from time import perf_counter

import numpy as np
import torch

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.base_worker import ActorWorker as BaseActorWorker
from roll.utils.functionals import adjust_sequence_length, masked_mean, agg_loss, compute_approx_kl
from roll.utils.train_infer_corrections import compute_train_infer_correction


ERPO_PROMPT_NLL_KEY = "erpo_prompt_nll"
ERPO_PROMPT_WEIGHT_KEY = "erpo_prompt_weight"


def normalize_prompt_nll_weights(
    prompt_nll: torch.Tensor,
    valid_prompt: torch.Tensor,
    batch_mean_nll: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize detached prompt NLLs into ERPO sample weights.

    ``batch_mean_nll`` may be supplied from a data-parallel reduction.  When
    omitted, the mean is computed from the local tensor, which is useful for
    standalone callers and unit tests.
    """
    prompt_nll = prompt_nll.detach().float()
    valid_prompt = valid_prompt.bool()
    weights = torch.ones_like(prompt_nll)
    if valid_prompt.any():
        if batch_mean_nll is None:
            batch_mean_nll = prompt_nll[valid_prompt].mean()
        weights[valid_prompt] = batch_mean_nll.detach().float() / prompt_nll[valid_prompt].clamp_min(eps)
    return weights.detach()


def compute_dynamic_prompt_logp_weights(
    log_probs: torch.Tensor,
    prompt_mask: torch.Tensor,
    eps: float = 1e-8,
    batch_mean_nll: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the online prompt weights used by the ERPO implementation.

    The original implementation uses the absolute sum of the current actor's
    prompt log probabilities as a per-prompt NLL proxy, then applies
    ``mean(prompt_nll) / prompt_nll`` to the policy loss.  The result is
    detached so prompt tokens only determine the response-loss scale.

    Samples without prompt tokens keep unit weight.  This is a defensive case
    for malformed batches; regular RLVR batches always contain prompt tokens.
    """
    prompt_mask = prompt_mask.to(dtype=log_probs.dtype)
    prompt_nll = (log_probs * prompt_mask).sum(dim=-1).abs()
    valid_prompt = prompt_mask.sum(dim=-1) > 0

    weights = normalize_prompt_nll_weights(
        prompt_nll=prompt_nll,
        valid_prompt=valid_prompt,
        batch_mean_nll=batch_mean_nll,
        eps=eps,
    )

    return weights.detach(), prompt_nll.detach()


def build_prompt_only_batch(data: DataProto) -> DataProto:
    """Build a compact right-padded prompt-only batch from an RLVR batch."""
    if "prompt_mask" in data.batch:
        prompt_mask = data.batch["prompt_mask"].bool()
    else:
        prompt_mask = data.batch["attention_mask"].bool() & ~data.batch["response_mask"].bool()

    prompt_lengths = prompt_mask.sum(dim=-1)
    if torch.any(prompt_lengths < 2):
        raise ValueError("dynamic prompt weighting requires at least two prompt tokens per sample")

    max_prompt_length = int(prompt_lengths.max().item())
    input_ids = data.batch["input_ids"][:, :max_prompt_length]
    attention_mask = prompt_mask[:, :max_prompt_length].long()

    expected_prefix_mask = (
        torch.arange(max_prompt_length, device=attention_mask.device).unsqueeze(0)
        < prompt_lengths.unsqueeze(1)
    )
    if not torch.equal(attention_mask.bool(), expected_prefix_mask):
        raise ValueError("dynamic prompt weighting expects right-padded prompt tokens to form a contiguous prefix")

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": torch.zeros_like(attention_mask),
    }
    if "position_ids" in data.batch:
        tensors["position_ids"] = data.batch["position_ids"][..., :max_prompt_length]

    return DataProto.from_dict(
        tensors=tensors,
        meta_info={
            "loss_mask_keys": [],
            "output_on_all_tp_cp_ranks": True,
        },
    )


def pad_prompt_batch_for_sequence_packing(prompt_batch: DataProto, sequence_length: int) -> DataProto:
    """Restore the configured tensor width before Megatron sequence packing.

    The packing loss wrapper identifies sequence dimensions using the configured
    strategy sequence length.  A compact prompt-only batch therefore needs to
    be padded back to that width before packing; the attention mask still makes
    packing remove these padding tokens, so model compute remains prompt-only.
    """
    prompt_sequence_length = prompt_batch.batch["input_ids"].size(-1)
    if prompt_sequence_length > sequence_length:
        raise ValueError(
            f"prompt sequence length {prompt_sequence_length} exceeds strategy sequence length {sequence_length}"
        )

    for key, value in prompt_batch.batch.items():
        prompt_batch.batch[key] = adjust_sequence_length(
            value,
            target_length=sequence_length,
            origin_seq_len=prompt_sequence_length,
            pad_value=0,
        )
    return prompt_batch


class ActorWorker(BaseActorWorker):

    def forward_func_prompt_nll(self, data: DataProto, output_tensor: torch.Tensor):
        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor,
            input_ids=data.batch["input_ids"],
            attention_mask=data.batch["attention_mask"],
        )
        prompt_mask = data.batch["attention_mask"][:, 1:].bool()
        prompt_nll = -(log_probs.float() * prompt_mask).sum(dim=-1)
        valid_prompt = prompt_mask.any(dim=-1)
        return torch.zeros((), device=output_tensor.device), {
            "prompt_nll": prompt_nll.detach(),
            "valid_prompt": valid_prompt.detach(),
        }

    def prepare_backward_batch(self, data: DataProto) -> dict[str, float]:
        """Precompute ERPO weights over the actual global optimizer batch."""
        if not self.pipeline_config.dynamic_prompt_logp_loss_weight:
            return {}

        start = perf_counter()
        prompt_batch = build_prompt_only_batch(data)
        if getattr(self.strategy, "use_sequence_packing", False):
            prompt_batch = pad_prompt_batch_for_sequence_packing(
                prompt_batch,
                sequence_length=self.strategy.seq_length,
            )
        prompt_batch.meta_info["micro_batch_size"] = self.worker_config.infer_batch_size

        with torch.no_grad():
            results = self.strategy.forward_step(
                batch=prompt_batch,
                forward_func=self.forward_func_prompt_nll,
            )
        if results is None:
            raise RuntimeError("prompt NLL precompute did not return results on this actor rank")

        prompt_nll = results["prompt_nll"].reshape(-1).float()
        valid_prompt = results["valid_prompt"].reshape(-1).bool()
        if prompt_nll.numel() != len(data):
            raise RuntimeError(
                f"prompt NLL result size {prompt_nll.numel()} does not match optimizer batch size {len(data)}"
            )

        global_stats = torch.stack(
            [
                prompt_nll[valid_prompt].sum(),
                valid_prompt.sum().to(dtype=prompt_nll.dtype),
            ]
        )
        self.strategy.op_data_parallel_sum(global_stats)
        if global_stats[1].item() <= 0:
            raise RuntimeError("dynamic prompt weighting found no valid prompts in the global optimizer batch")
        global_mean_nll = global_stats[0] / global_stats[1]
        prompt_weights = normalize_prompt_nll_weights(
            prompt_nll=prompt_nll,
            valid_prompt=valid_prompt,
            batch_mean_nll=global_mean_nll,
        )

        data.batch[ERPO_PROMPT_NLL_KEY] = prompt_nll
        data.batch[ERPO_PROMPT_WEIGHT_KEY] = prompt_weights
        return {
            "time/actor_train/prompt_weight_precompute": perf_counter() - start,
            "actor/prompt_weight_global_mean_nll": global_mean_nll.item(),
            "actor/prompt_weight_global_samples": global_stats[1].item(),
        }

    def loss_func(self, data: DataProto, output_tensor: torch.Tensor):
        """
        loss func接口定义:
            data: DataProto, 由train_step透传
            output_tensor: torch.Tensor, model.forward()的输出Tensor
        """
        response_mask = data.batch["response_mask"][:, 1:].long()
        final_response_mask = data.batch.get("final_response_mask", response_mask)
        ref_log_probs = data.batch["ref_log_probs"]
        advantages = data.batch["advantages"]

        batch_num_tokens = data.meta_info['batch_num_tokens']
        global_valid_samples = data.meta_info['global_valid_samples']
        if 'final_response_mask' not in batch_num_tokens:
            batch_num_tokens['final_response_mask'] = batch_num_tokens['response_mask']
            global_valid_samples['final_response_mask'] = global_valid_samples['response_mask']

        log_probs = self.strategy.op_compute_log_probs(
            logits=output_tensor, input_ids=data.batch["input_ids"], attention_mask=data.batch["attention_mask"]
        )
        old_log_probs = self.get_old_log_probs_with_cache(data, log_probs)
        infer_log_probs = data.batch.get("infer_logprobs", old_log_probs)
        infer_log_probs = infer_log_probs if len(infer_log_probs) > 0 else old_log_probs

        train_infer_metric = {}
        if not self.pipeline_config.enable_old_logprobs_recompute:
            train_infer_is_weight, filter_mask, train_infer_metric = compute_train_infer_correction(
                cfg=self.pipeline_config.train_infer_correction,
                response_mask=response_mask,
                old_log_probs=old_log_probs,
                infer_log_probs=infer_log_probs,
                global_valid_samples=global_valid_samples['response_mask'],
                global_valid_tokens=batch_num_tokens['response_mask'],
            )

            # Apply filter mask to both response_mask and final_response_mask
            response_mask = response_mask.long() * filter_mask.long()
            final_response_mask = final_response_mask.long() * filter_mask.long()
        else:
            train_infer_is_weight = data.batch['train_infer_is_weight']

        attention_mask_shifted = data.batch["attention_mask"][:, 1:].long()
        # Keep prompt membership independent of response filtering. A filtered
        # response token must never be reclassified as a prompt token.
        original_response_mask = data.batch["response_mask"][:, 1:].long()
        prompt_kl_mask = attention_mask_shifted * (1 - original_response_mask)
        response_kl_mask = final_response_mask
        all_kl_mask = attention_mask_shifted

        valid_samples = torch.any(final_response_mask > 0, dim=1).float()
        sample_weights = self.compute_sample_weights(data, response_mask)
        prompt_nll = None
        if self.pipeline_config.dynamic_prompt_logp_loss_weight:
            if ERPO_PROMPT_WEIGHT_KEY not in data.batch or ERPO_PROMPT_NLL_KEY not in data.batch:
                raise RuntimeError(
                    "dynamic prompt weights must be precomputed for the full optimizer batch before packing"
                )
            prompt_weights = data.batch[ERPO_PROMPT_WEIGHT_KEY].float()
            prompt_nll = data.batch[ERPO_PROMPT_NLL_KEY].float()
            sample_weights = sample_weights * prompt_weights

        # Select the tokens used to compute the reference-model KL loss.
        if self.pipeline_config.kl_loss_mask_mode == "response":
            kl_mask = response_kl_mask
        elif self.pipeline_config.kl_loss_mask_mode == "prompt":
            kl_mask = prompt_kl_mask
        elif self.pipeline_config.kl_loss_mask_mode == "all":
            kl_mask = all_kl_mask
        else:
            raise ValueError(f"Invalid kl_loss_mask_mode: {self.pipeline_config.kl_loss_mask_mode}")

        ref_kl = compute_approx_kl(log_probs=log_probs, log_probs_base=ref_log_probs, kl_penalty="k3")
        kl_loss = agg_loss(loss_mat=ref_kl,
                         loss_mask=kl_mask,
                         loss_agg_mode=self.pipeline_config.loss_agg_mode,
                         global_valid_samples=global_valid_samples['final_response_mask'],)
        kl_metric = {
            "actor/all_kl@sum": agg_loss(
                loss_mat=ref_kl, loss_mask=all_kl_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                global_valid_samples=global_valid_samples['final_response_mask'],
            ).detach().item(),
            "actor/prompt_kl@sum": agg_loss(
                loss_mat=ref_kl, loss_mask=prompt_kl_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                global_valid_samples=global_valid_samples['final_response_mask'],
            ).detach().item(),
            "actor/response_kl@sum": agg_loss(
                loss_mat=ref_kl, loss_mask=response_kl_mask,
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                global_valid_samples=global_valid_samples['final_response_mask'],
            ).detach().item(),
        }

        approxkl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="mse"
        )
        policykl = compute_approx_kl(
            log_probs=log_probs, log_probs_base=old_log_probs, action_mask=response_mask, kl_penalty="kl"
        )

        if self.pipeline_config.importance_sampling == "token":
            ratio = (log_probs - old_log_probs).exp()
        elif self.pipeline_config.importance_sampling == "seq":
            log_ratio = log_probs - old_log_probs
            masked_log_ratio = masked_mean(log_ratio, final_response_mask, dim=-1)
            ratio = masked_log_ratio.exp().unsqueeze(-1).expand_as(log_ratio)

        pg_clip_low = self.pipeline_config.pg_clip_low if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        pg_clip_high = self.pipeline_config.pg_clip_high if self.pipeline_config.use_pg_clip_range else self.pipeline_config.pg_clip
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - pg_clip_low, 1 + pg_clip_high) * advantages

        loss = -torch.min(surr1, surr2)

        if self.pipeline_config.dual_clip_loss:
            dual_clip_loss = -torch.max(-loss, (1 + self.pipeline_config.pg_clip * 2) * advantages)
            loss = torch.where(advantages < 0, dual_clip_loss, loss)

        if self.pipeline_config.train_infer_correction.is_weight.enabled:
            loss = loss * train_infer_is_weight

        weighted_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    weights=sample_weights,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)
        original_pg_loss = agg_loss(loss_mat=loss, loss_mask=final_response_mask,
                                    loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                    batch_num_tokens=batch_num_tokens['final_response_mask'],
                                    global_valid_samples=global_valid_samples['final_response_mask'],)

        clipped_low = (ratio < 1 - pg_clip_low).float()
        clipped_high = (ratio > 1 + pg_clip_high).float()
        clipped = (clipped_low + clipped_high).float()

        if self.pipeline_config.use_kl_loss:
            total_loss = weighted_pg_loss + kl_loss * self.pipeline_config.kl_loss_coef
        else:
            total_loss = weighted_pg_loss

        total_loss = total_loss * self.pipeline_config.rl_loss_coef

        if self.pipeline_config.entropy_loss_coef > 0:
            entropy = self.strategy.op_compute_entropy(logits=output_tensor, attention_mask=data.batch["response_mask"])
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=data.batch["response_mask"][:, 1:],
                loss_agg_mode=self.pipeline_config.loss_agg_mode,
                batch_num_tokens=batch_num_tokens['response_mask'],
                global_valid_samples=global_valid_samples['response_mask'],
            )
            total_loss = total_loss - entropy_loss * self.pipeline_config.entropy_loss_coef

        loss_metric = {
            "actor/ppo_ratio_high_clipfrac@sum": agg_loss(loss_mat=clipped_high, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_low_clipfrac@sum": agg_loss(loss_mat=clipped_low, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ppo_ratio_clipfrac@sum": agg_loss(loss_mat=clipped, loss_mask=final_response_mask,
                                loss_agg_mode='token-mean',
                                 batch_num_tokens=batch_num_tokens['final_response_mask']).detach().item(),
            "actor/ratio_mean@sum": agg_loss(loss_mat=ratio, loss_mask=response_mask,
                                loss_agg_mode='seq-mean-token-mean',
                                 global_valid_samples=global_valid_samples['response_mask']).detach().item(),
            "actor/ratio_max@max": torch.max(ratio * response_mask).detach().item(),
            "actor/ratio_min@min": torch.min(ratio * response_mask + (1 - response_mask) * 1e10).detach().item(),
            "actor/clipfrac@sum": agg_loss(loss_mat=torch.lt(surr2, surr1).float(), loss_mask=response_mask,
                                loss_agg_mode=self.pipeline_config.loss_agg_mode, batch_num_tokens=batch_num_tokens['final_response_mask'],
                                       global_valid_samples=global_valid_samples['response_mask']).detach().item(),
        }

        pg_metrics = {
            "actor/pg_loss@sum": original_pg_loss.detach().item(),
            "actor/weighted_pg_loss@sum": weighted_pg_loss.detach().item(),
            "actor/kl_loss@sum": kl_loss.detach().item(),
            "actor/total_loss@sum": total_loss.detach().item(),
            "actor/approxkl@sum": agg_loss(loss_mat=approxkl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens['response_mask'],
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/policykl@sum": agg_loss(loss_mat=policykl, loss_mask=response_mask,
                                       loss_agg_mode=self.pipeline_config.loss_agg_mode,
                                       batch_num_tokens=batch_num_tokens['response_mask'],
                                        global_valid_samples=global_valid_samples['response_mask'],).detach().item(),
            "actor/valid_samples@sum": valid_samples.sum().detach().item(),
            "actor/total_samples@sum": float(valid_samples.size(0)),
            "actor/valid_sample_ratio@sum": (valid_samples.sum() / global_valid_samples['response_mask']).detach().item(),
            "actor/sample_weights_mean@mean": sample_weights.mean().detach().item(),
            "actor/sample_weights_min@min": sample_weights.min().detach().item(),
            "actor/sample_weights_max@max": sample_weights.max().detach().item(),
            **kl_metric,
            **loss_metric,
            **train_infer_metric,
        }
        if prompt_nll is not None:
            pg_metrics.update(
                {
                    "actor/prompt_nll_mean": prompt_nll.mean().item(),
                    "actor/prompt_nll_min": prompt_nll.min().item(),
                    "actor/prompt_nll_max": prompt_nll.max().item(),
                }
            )

        return total_loss, pg_metrics

    def compute_sample_weights(self, data: DataProto, response_mask: torch.Tensor):
        """
        可以基于难度和长度的样本权重
        """
        batch_size = response_mask.shape[0]
        sample_weights = torch.ones(batch_size, device=response_mask.device)

        # 1. 基于难度的权重 - 例如：难度越高，权重越大
        if self.pipeline_config.difficulty_loss_weight and "difficulty" in data.non_tensor_batch:
            try:
                difficulty = data.non_tensor_batch["difficulty"]
                if isinstance(difficulty, np.ndarray):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                elif not isinstance(difficulty, torch.Tensor):
                    difficulty = torch.tensor(difficulty, dtype=torch.float32, device=response_mask.device)
                norm_difficulty = torch.clamp(difficulty, 0.0, 1.0)
                difficulty_weights = 0.5 + 1.5 * norm_difficulty
                sample_weights = sample_weights * difficulty_weights
            except Exception as e:
                self.logger.warning(f"跳过difficulty权重计算：{str(e)}")

        # 2. 基于长度的权重 - 例如：长度越长，权重越小
        response_lengths = response_mask.sum(dim=1).float()
        if self.pipeline_config.length_loss_weight:
            # 同样归一化长度到[0.5, 2.0]范围
            norm_lengths = (response_lengths - response_lengths.min()) / (
                    response_lengths.max() - response_lengths.min() + 1e-8
            )
            length_weights = 1.5 - norm_lengths
            sample_weights = sample_weights * length_weights

        if sample_weights.sum() > 0:
            sample_weights = sample_weights * (batch_size / (sample_weights.sum() + 1e-8))

        return sample_weights
