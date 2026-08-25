from pathlib import Path

import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "erpo"
    / "erpo_base_qwen25_7b.yaml"
)


def test_erpo_parallelism_preserves_batches():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    num_nodes = 2
    total_gpus = num_nodes * config["num_gpus_per_node"]
    actor = config["actor_train"]
    reference = config["reference"]
    actor_tp = actor["strategy_args"]["strategy_config"]["tensor_model_parallel_size"]
    reference_tp = reference["strategy_args"]["strategy_config"]["tensor_model_parallel_size"]
    actor_dp = len(range(0, total_gpus)) // actor_tp

    assert actor_tp == 1
    assert reference_tp == 1
    assert total_gpus == 16
    assert actor_dp == 16

    micro_batch = actor["training_args"]["per_device_train_batch_size"]
    grad_accumulation = actor["training_args"]["gradient_accumulation_steps"]
    global_actor_batch = actor_dp * micro_batch * grad_accumulation
    # actor.global_batch_size is expressed in prompt groups and is
    # multiplied by rollout.n before response-level actor updates.
    erpo_response_global_batch = 128 * config["num_return_sequences_in_group"]
    assert global_actor_batch == erpo_response_global_batch == 1024

    rollout_size = config["rollout_batch_size"] * config["num_return_sequences_in_group"]
    assert rollout_size == 1024
    assert rollout_size // global_actor_batch == 1
    assert config["max_steps"] * (rollout_size // global_actor_batch) == 960

    expected_mapping = "list(range(0,16))"
    assert actor["device_mapping"] == expected_mapping
    assert config["actor_infer"]["device_mapping"] == expected_mapping
    assert reference["device_mapping"] == expected_mapping

    # These correspond to the per-device experience/log-prob micro-batch.
    assert actor["infer_batch_size"] == 4
    assert reference["infer_batch_size"] == 4


def test_erpo_packs_sequences_without_changing_optimizer_batches():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    actor = config["actor_train"]
    reference = config["reference"]
    sequence_length = config["sequence_length"]

    # Dynamic batching groups a variable number of samples per micro-batch and
    # can therefore change the optimizer-step boundary. Sequence packing runs
    # inside the existing 1024-response optimizer batch instead.
    assert actor.get("use_dynamic_batching_in_train", False) is False
    assert actor.get("use_dynamic_batching_in_infer", False) is False

    for worker in (actor, reference):
        assert worker["use_sequence_packing"] is True
        packing = worker["sequence_packing_args"]
        assert packing["algorithm"] == "load_balance"
        assert packing["max_packed_sequence_length_train"] == "${sequence_length}"
        assert packing["max_packed_sequence_length_forward"] == "${sequence_length}"
        assert packing["min_num_micro_batches_train"] == 1
        assert packing["min_num_micro_batches_forward"] == 1

    # Hydra resolves both interpolation values to the fixed model limit. The
    # packed ceiling must be able to contain the longest individual sequence.
    assert sequence_length == 9216


def test_erpo_uses_8k_response_budget():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["prompt_length"] == 1024
    assert config["response_length"] == 8192
    assert config["sequence_length"] == 9216
    assert (
        config["actor_infer"]["generating_args"]["max_new_tokens"]
        == "${response_length}"
    )
    assert (
        config["validation"]["generating_args"]["max_new_tokens"]
        == "${response_length}"
    )
    assert (
        config["actor_infer"]["strategy_args"]["strategy_config"]["max_model_len"]
        == "${sequence_length}"
    )


def test_erpo_tracks_with_swanlab():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["track_with"] == "swanlab"
    assert config["tracker_kwargs"]["project"] == "erpo"
    assert config["tracker_kwargs"]["experiment_name"] == "${exp_name}"
    assert config["exp_name"].startswith("${oc.env:RUN_NAME,")
