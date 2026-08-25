from types import SimpleNamespace

import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.rlvr.actor_worker import (
    ERPO_PROMPT_NLL_KEY,
    ERPO_PROMPT_WEIGHT_KEY,
    ActorWorker,
    build_prompt_only_batch,
    compute_dynamic_prompt_logp_weights,
    normalize_prompt_nll_weights,
    pad_prompt_batch_for_sequence_packing,
)


def test_dynamic_prompt_weights_match_erpo_formula():
    log_probs = torch.tensor(
        [
            [-1.0, -2.0, -3.0],
            [-2.0, -4.0, -6.0],
        ],
        requires_grad=True,
    )
    prompt_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
        ]
    )

    weights, prompt_nll = compute_dynamic_prompt_logp_weights(log_probs, prompt_mask)

    torch.testing.assert_close(prompt_nll, torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(weights, torch.tensor([1.5, 0.75]))
    assert not weights.requires_grad
    assert not prompt_nll.requires_grad


def test_dynamic_prompt_weights_ignore_padding_and_handle_empty_prompt():
    log_probs = torch.tensor(
        [
            [-1.0, -2.0, -100.0],
            [-3.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ]
    )
    prompt_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    weights, prompt_nll = compute_dynamic_prompt_logp_weights(log_probs, prompt_mask)

    torch.testing.assert_close(prompt_nll, torch.tensor([3.0, 3.0, 0.0]))
    torch.testing.assert_close(weights, torch.ones(3))


def test_dynamic_prompt_weights_use_global_optimizer_batch_mean():
    local_prompt_nll = torch.tensor([2.0, 4.0])
    valid_prompt = torch.tensor([True, True])

    weights = normalize_prompt_nll_weights(
        prompt_nll=local_prompt_nll,
        valid_prompt=valid_prompt,
        batch_mean_nll=torch.tensor(5.0),
    )

    torch.testing.assert_close(weights, torch.tensor([2.5, 1.25]))


def test_build_prompt_only_batch_removes_responses_and_padding():
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10, 11, 12, 20, 21, 0], [30, 31, 40, 0, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]]),
            "prompt_mask": torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]),
            "response_mask": torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 0], [0, 1, 2, 0, 0, 0]]),
        }
    )

    prompt_batch = build_prompt_only_batch(data)

    torch.testing.assert_close(prompt_batch.batch["input_ids"], torch.tensor([[10, 11, 12], [30, 31, 40]]))
    torch.testing.assert_close(prompt_batch.batch["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    torch.testing.assert_close(prompt_batch.batch["response_mask"], torch.zeros(2, 3, dtype=torch.long))
    torch.testing.assert_close(prompt_batch.batch["position_ids"], torch.tensor([[0, 1, 2], [0, 1, 2]]))


def test_pad_prompt_batch_for_sequence_packing_restores_strategy_width():
    prompt_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10, 11, 12], [30, 31, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "response_mask": torch.zeros(2, 3, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        }
    )

    padded = pad_prompt_batch_for_sequence_packing(prompt_batch, sequence_length=8)

    assert all(value.size(-1) == 8 for value in padded.batch.values())
    torch.testing.assert_close(padded.batch["input_ids"][:, :3], torch.tensor([[10, 11, 12], [30, 31, 0]]))
    torch.testing.assert_close(padded.batch["attention_mask"][:, 3:], torch.zeros(2, 5, dtype=torch.long))


class _FakePromptWeightStrategy:
    use_sequence_packing = True
    seq_length = 8

    def forward_step(self, batch, forward_func):
        assert batch.batch["input_ids"].shape == (2, 8)
        assert batch.batch["attention_mask"].sum(dim=-1).tolist() == [3, 2]
        return {
            "prompt_nll": torch.tensor([2.0, 4.0]),
            "valid_prompt": torch.tensor([True, True]),
        }

    def op_data_parallel_sum(self, stats):
        # Simulate a second DP rank with prompt NLLs 4 and 10.
        stats += torch.tensor([14.0, 2.0])
        return stats


def test_prepare_backward_batch_attaches_global_weights_before_packing():
    data = DataProto.from_dict(
        tensors={
            "input_ids": torch.tensor([[10, 11, 12, 20], [30, 31, 40, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "prompt_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "response_mask": torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]]),
        }
    )
    worker = ActorWorker.__new__(ActorWorker)
    worker.pipeline_config = SimpleNamespace(dynamic_prompt_logp_loss_weight=True)
    worker.worker_config = SimpleNamespace(infer_batch_size=4)
    worker.strategy = _FakePromptWeightStrategy()

    metrics = worker.prepare_backward_batch(data)

    # Global mean is (2 + 4 + 4 + 10) / 4 = 5.
    torch.testing.assert_close(data.batch[ERPO_PROMPT_NLL_KEY], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(data.batch[ERPO_PROMPT_WEIGHT_KEY], torch.tensor([2.5, 1.25]))
    assert metrics["actor/prompt_weight_global_mean_nll"] == 5.0
    assert metrics["actor/prompt_weight_global_samples"] == 4.0
