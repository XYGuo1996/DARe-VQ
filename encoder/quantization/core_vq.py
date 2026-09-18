# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

from .. import distrib


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_rank0() -> bool:
    return (not _dist_is_initialized()) or torch.distributed.get_rank() == 0


def _all_reduce_inplace(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        torch.distributed.all_reduce(tensor)
    return tensor


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


@torch.no_grad()
def kmeans(samples, num_clusters: int, num_iters: int = 10, batch_size: int = 1024):
    """Batched k-means with FP32 accumulation to prevent overflow on large codebooks."""
    dim, dtype = samples.shape[-1], samples.dtype
    device = samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        new_means = torch.zeros((num_clusters, dim), dtype=torch.float32, device=device)
        bins = torch.zeros(num_clusters, dtype=torch.long, device=device)

        num_samples = samples.shape[0]
        bs = batch_size if batch_size > 0 else num_samples

        for i in range(0, num_samples, bs):
            sample_chunk = samples[i : i + bs]

            sample_chunk_f32 = sample_chunk.to(torch.float32)
            means_f32 = means.to(torch.float32)

            x_sq = (sample_chunk_f32 ** 2).sum(dim=1, keepdim=True)
            y_sq = (means_f32 ** 2).sum(dim=1)
            xy = sample_chunk_f32 @ means_f32.T

            dists = x_sq + y_sq - 2 * xy
            buckets = dists.argmin(dim=-1)

            bins += torch.bincount(buckets, minlength=num_clusters)
            new_means.index_add_(0, buckets, sample_chunk_f32)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = new_means / bins_min_clamped.unsqueeze(-1)
        new_means = new_means.to(dtype)

        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins

def _sync_n_q_across_ranks(
    n_q: int,
    device: torch.device,
) -> int:
    """Use rank 0's active quantizer count on every rank."""
    if not _dist_is_initialized():
        return int(n_q)

    n_q_tensor = torch.tensor(
        [int(n_q)],
        dtype=torch.long,
        device=device,
    )

    torch.distributed.broadcast(
        n_q_tensor,
        src=0,
    )

    return int(n_q_tensor.item())

class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
        ema_update_every: int = 1,
        dead_code_strategy: str = "random",
        split_candidate_strategy: str = "frequency",
        split_frequency_weight: float = 1.0,
        split_variance_weight: float = 1.0,
        spectokenizer_bank_size: int = 32768,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.ema_update_every = max(1, ema_update_every)
        self.dead_code_strategy = dead_code_strategy
        self.split_candidate_strategy = split_candidate_strategy
        self.split_frequency_weight = split_frequency_weight
        self.split_variance_weight = split_variance_weight

        self._use_delayed_ema = self.ema_update_every > 1
        self._use_split = self.dead_code_strategy == "split"
        self._use_spectokenizer = self.dead_code_strategy == "spectokenizer"
        self._use_history_reservoir = self._use_split or self._use_spectokenizer
        self.spectokenizer_bank_size = max(
            32768,
            self.codebook_size,
            int(spectokenizer_bank_size),
        )

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        if self._use_delayed_ema:
            self.register_buffer('pending_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('pending_embed_sum', torch.zeros_like(self.embed))
            self.register_buffer('pending_steps', torch.zeros(1, dtype=torch.long))

        if self._use_history_reservoir:
            self.reservoir = torch.empty(0, dim)
            self.reservoir_filled = 0
            self.reservoir_seen = 0
            
        if self._use_split:
            self._pending_samples_cpu: tp.List[torch.Tensor] = []
            self._pending_indices_cpu: tp.List[torch.Tensor] = []
            self._pending_copy_events: tp.List[torch.cuda.Event] = []

        self._pending_copy_events = []

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # distrib.broadcast_tensors(self.buffers())
        self._broadcast_codebook_state()

    def _broadcast_codebook_state(self):
        """Broadcast only the persistent codebook states from rank 0."""
        if not _dist_is_initialized():
            return

        torch.distributed.broadcast(self.inited, src=0)
        torch.distributed.broadcast(self.cluster_size, src=0)
        torch.distributed.broadcast(self.embed, src=0)
        torch.distributed.broadcast(self.embed_avg, src=0)

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

        if self._use_split:
            self.embed_avg.data.copy_(modified_codebook * self.cluster_size.unsqueeze(1))
            if torch.any(mask):
                mean_cluster_size = self.cluster_size[~mask].mean()
                if torch.isnan(mean_cluster_size) or mean_cluster_size == 0:
                    mean_cluster_size = torch.tensor(1.0, device=self.cluster_size.device)
                self.cluster_size.data[mask] = mean_cluster_size
                self.embed_avg.data[mask] = modified_codebook[mask] * mean_cluster_size
        elif torch.any(mask):
            mean_cluster_size = self.cluster_size[~mask].mean()
            if torch.isnan(mean_cluster_size) or mean_cluster_size == 0:
                mean_cluster_size = torch.tensor(1.0, device=self.cluster_size.device)
            self.cluster_size.data[mask] = mean_cluster_size
            self.embed_avg.data[mask] = modified_codebook[mask] * mean_cluster_size

    def expire_codes_(self, batch_samples, current_counts=None):
        if self.threshold_ema_dead_code == 0:
            return 0

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return 0
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())
        return expired_codes.sum().item()

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind_out = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind_out)

        if self.training:
            counts = embed_onehot.sum(0)
            embed_sum = (x.t() @ embed_onehot).t()

            if self._use_delayed_ema:
                self._accumulate_or_update(counts, embed_sum, x, embed_ind)
            else:
                self._immediate_update(counts, embed_sum, x, embed_ind)

        return quantize, embed_ind_out

    def _immediate_update(
        self,
        counts,
        embed_sum,
        x,
        embed_ind,
    ):
        counts, embed_sum = self._sync_ema_statistics(
            counts,
            embed_sum,
        )

        n_replaced = 0

        if self._use_split:
            if _is_rank0():
                self._update_reservoir(x)

                self._pending_samples_cpu.append(
                    self._async_copy_to_pinned_cpu(x)
                )
                self._pending_indices_cpu.append(
                    self._async_copy_to_pinned_cpu(embed_ind)
                )

                n_replaced = self._expire_codes_split(counts)

            self._broadcast_codebook_state()

            if _is_rank0() and n_replaced > 0:
                print(
                    f"Replaced {n_replaced} cluster centers.",
                    flush=True,
                )

        elif self._use_spectokenizer:
            self._update_reservoir(x)
            n_replaced = self._expire_codes_spectokenizer(x)

            if _is_rank0() and n_replaced > 0:
                print(
                    f"Replaced {n_replaced} dead codes from reservoir.",
                    flush=True,
                )

        else:
            self.expire_codes_(x)

        ema_inplace(
            self.cluster_size,
            counts,
            self.decay,
        )
        ema_inplace(
            self.embed_avg,
            embed_sum,
            self.decay,
        )

        cluster_size = (
            laplace_smoothing(
                self.cluster_size,
                self.codebook_size,
                self.epsilon,
            )
            * self.cluster_size.sum()
        )

        embed_normalized = (
            self.embed_avg
            / cluster_size.unsqueeze(1)
        )

        self.embed.data.copy_(embed_normalized)

    def _async_copy_to_pinned_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_cuda:
            return tensor.detach().cpu()

        cpu_tensor = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )

        cpu_tensor.copy_(tensor.detach(), non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._pending_copy_events.append(event)

        return cpu_tensor
    
    def _accumulate_or_update(self, counts, embed_sum, x, embed_ind):
        counts = counts.detach()
        embed_sum = embed_sum.detach()

        if self._use_split and x is not None and _is_rank0():
            # Maintain a SpecTokenizer-style historical reservoir for fallback,
            # while pending samples remain the source for split statistics.
            self._update_reservoir(x)
            self._pending_samples_cpu.append(self._async_copy_to_pinned_cpu(x))
            self._pending_indices_cpu.append(self._async_copy_to_pinned_cpu(embed_ind))

        if self._use_spectokenizer and x is not None:
            self._update_reservoir(x)

        self.pending_cluster_size.add_(counts)
        self.pending_embed_sum.add_(embed_sum)
        self.pending_steps += 1

        if self.pending_steps.item() >= self.ema_update_every:
            self._apply_delayed_ema_update(self.pending_cluster_size, self.pending_embed_sum, x)
            self._reset_pending()

    def _apply_delayed_ema_update(
        self,
        counts,
        embed_sum,
        x,
    ):
        if _is_rank0():
            print(
                "EMA update performed.",
                flush=True,
            )

        counts, embed_sum = self._sync_ema_statistics(
            counts,
            embed_sum,
        )

        n_replaced = 0

        if self._use_split:
            if _is_rank0():
                n_replaced = self._expire_codes_split(counts)

            self._broadcast_codebook_state()

        elif self._use_spectokenizer:
            n_replaced = self._expire_codes_spectokenizer(
                x.detach() if x is not None else None
            )

        else:
            if x is not None:
                self.expire_codes_(x.detach())

        if _is_rank0():
            print(
                f"Replaced {n_replaced} cluster centers.",
                flush=True,
            )

        ema_inplace(
            self.cluster_size,
            counts,
            self.decay,
        )
        ema_inplace(
            self.embed_avg,
            embed_sum,
            self.decay,
        )

        cluster_size = (
            laplace_smoothing(
                self.cluster_size,
                self.codebook_size,
                self.epsilon,
            )
            * self.cluster_size.sum()
        )

        self.embed.data.copy_(
            self.embed_avg
            / cluster_size.unsqueeze(1)
        )

    def _sync_ema_statistics(self, counts, embed_sum):
        counts = counts.detach().contiguous()
        embed_sum = embed_sum.detach().contiguous()

        if _dist_is_initialized():
            torch.distributed.all_reduce(
                counts,
                op=torch.distributed.ReduceOp.SUM,
            )
            torch.distributed.all_reduce(
                embed_sum,
                op=torch.distributed.ReduceOp.SUM,
            )

        return counts, embed_sum

    def _expire_codes_split(self, current_counts):

        self._synchronize_pending_cpu_copies()

        if self.threshold_ema_dead_code == 0:
            self._pending_samples_cpu.clear()
            self._pending_indices_cpu.clear()
            return 0

        expired_codes = (
            self.cluster_size < self.threshold_ema_dead_code
        )

        expired_idx = expired_codes.nonzero(as_tuple=False).view(-1)
        n_expired = int(expired_idx.numel())

        if n_expired == 0:
            self._pending_samples_cpu.clear()
            self._pending_indices_cpu.clear()
            return 0

        if not self._pending_samples_cpu:
            self._pending_indices_cpu.clear()
            return 0

        (
            sorted_samples_cpu,
            sorted_indices_cpu,
            counts_cpu,
            offsets_cpu,
        ) = self._build_sorted_pending_cpu()

        candidate_codes = self._rank_split_candidates_fast_cpu(
            sorted_samples_cpu=sorted_samples_cpu,
            sorted_indices_cpu=sorted_indices_cpu,
            counts_cpu=counts_cpu,
            n_needed=n_expired,
        )

        n_split = min(len(candidate_codes), n_expired)
        n_replaced = 0
        fallback_dead_indices: tp.List[int] = []

        for split_code, dead_idx in zip(
            candidate_codes[:n_split],
            expired_idx[:n_split].tolist(),
        ):
            samples_for_split_cpu = self._get_sorted_samples_for_code_cpu(
                sorted_samples_cpu,
                offsets_cpu,
                split_code,
            )

            if (
                samples_for_split_cpu is None
                or samples_for_split_cpu.shape[0] < 2
            ):
                fallback_dead_indices.append(dead_idx)
                continue

            samples_for_split = samples_for_split_cpu.to(
                device=self.embed.device,
                dtype=self.embed.dtype,
            )

            new_means, _ = kmeans(
                samples_for_split,
                2,
                num_iters=10,
            )
            del samples_for_split

            self.embed.data[dead_idx].copy_(new_means[0])
            self.embed.data[split_code].copy_(new_means[1])

            old_size = self.cluster_size[split_code].clone()
            half_size = old_size / 2.0

            self.cluster_size[dead_idx] = half_size
            self.cluster_size[split_code] = half_size
            self.embed_avg[dead_idx] = self.embed[dead_idx] * half_size
            self.embed_avg[split_code] = self.embed[split_code] * half_size

            n_replaced += 1

        fallback_dead_indices.extend(
            expired_idx[n_split:].tolist()
        )
        if fallback_dead_indices:
            fallback_dead_idx = torch.tensor(
                fallback_dead_indices,
                dtype=torch.long,
                device=expired_idx.device,
            )
            n_replaced += self._random_replace_dead_indices_from_reservoir(
                fallback_dead_idx
            )

        self._pending_samples_cpu.clear()
        self._pending_indices_cpu.clear()

        return n_replaced

    def _random_replace_dead_indices_from_reservoir(
        self,
        dead_indices: torch.Tensor,
    ) -> int:
        if dead_indices.numel() == 0:
            return 0

        filled = int(getattr(self, "reservoir_filled", 0))
        if filled > 0:
            pool = self.reservoir[:filled]
        elif self._pending_samples_cpu:
            pool = torch.cat(self._pending_samples_cpu, dim=0)
        else:
            return 0

        if pool.shape[0] == 0:
            return 0

        dead_indices = dead_indices.to(
            device=self.embed.device,
            dtype=torch.long,
        )

        sample_idx = torch.randint(
            0,
            pool.shape[0],
            (dead_indices.numel(),),
            device=pool.device,
        )
        replacement_vectors = pool[sample_idx].to(
            device=self.embed.device,
            dtype=self.embed.dtype,
        )

        self.embed.data.index_copy_(
            0,
            dead_indices,
            replacement_vectors,
        )

        active_mask = self.cluster_size >= self.threshold_ema_dead_code
        if torch.any(active_mask):
            bootstrap_size = self.cluster_size[active_mask].mean()
        else:
            bootstrap_size = torch.tensor(
                1.0,
                device=self.cluster_size.device,
                dtype=self.cluster_size.dtype,
            )

        decay_safe_threshold = (
            float(self.threshold_ema_dead_code)
            / max(float(self.decay), 1e-8)
        )
        bootstrap_size = torch.clamp(
            bootstrap_size,
            min=decay_safe_threshold,
        )

        self.cluster_size[dead_indices] = bootstrap_size
        self.embed_avg[dead_indices] = (
            self.embed[dead_indices] * bootstrap_size
        )

        return int(dead_indices.numel())

    def _synchronize_pending_cpu_copies(self):
        if not self._use_split:
            return

        if not hasattr(self, "_pending_copy_events"):
            return

        for event in self._pending_copy_events:
            event.synchronize()

        self._pending_copy_events.clear()
        
    def _rank_split_candidates(self, all_indices_cpu: torch.Tensor) -> tp.List[int]:
        min_size = max(self.threshold_ema_dead_code * 2, self.cluster_size.mean() * 2)
        candidate_mask = self.cluster_size > 5
        candidate_indices = candidate_mask.nonzero(as_tuple=False).view(-1)
        if len(candidate_indices) == 0:
            return []

        if self.split_candidate_strategy == "frequency":
            sorted_idx = torch.argsort(self.cluster_size[candidate_indices], descending=True)
            return candidate_indices[sorted_idx].tolist()

        freq_scores = self.cluster_size[candidate_indices].float()

        var_scores = torch.zeros(len(candidate_indices), device=self.embed.device)
        for i, code_idx in enumerate(candidate_indices.tolist()):
            mask_cpu = all_indices_cpu == code_idx
            n_samples = mask_cpu.sum().item()
            if n_samples < 2:
                var_scores[i] = 0.0
                continue
            samples = self._gather_samples_by_mask(mask_cpu)
            centroid = self.embed[code_idx].cpu()
            dists = ((samples - centroid) ** 2).sum(dim=-1)
            var_scores[i] = dists.mean().item()

        if self.split_candidate_strategy == "frequency_variance":
            norm_freq = freq_scores / (freq_scores.max() + 1e-8)
            norm_var = var_scores / (var_scores.max() + 1e-8)
            combined = (self.split_frequency_weight * norm_freq +
                        self.split_variance_weight * norm_var)
        else:
            combined = (freq_scores ** self.split_frequency_weight *
                        var_scores ** self.split_variance_weight)

        sorted_idx = torch.argsort(combined, descending=True)
        return candidate_indices[sorted_idx].tolist()

    def _gather_samples_by_mask(self, mask_cpu: torch.Tensor) -> torch.Tensor:
        chunks = []
        offset = 0
        for samples_chunk in self._pending_samples_cpu:
            chunk_len = samples_chunk.shape[0]
            chunk_mask = mask_cpu[offset : offset + chunk_len]
            if chunk_mask.any():
                chunks.append(samples_chunk[chunk_mask])
            offset += chunk_len
        return torch.cat(chunks, dim=0)

    def _get_random_sample_from_cpu(self) -> tp.Optional[torch.Tensor]:
        total = sum(s.shape[0] for s in self._pending_samples_cpu)
        if total == 0:
            return None
        idx = torch.randint(0, total, (1,)).item()
        offset = 0
        for samples_chunk in self._pending_samples_cpu:
            chunk_len = samples_chunk.shape[0]
            if offset + chunk_len > idx:
                return samples_chunk[idx - offset]
            offset += chunk_len
        return self._pending_samples_cpu[-1][-1]

    @torch.no_grad()
    def _update_reservoir(self, samples):
        samples = rearrange(samples, "... d -> (...) d").detach()

        size = self.spectokenizer_bank_size

        if self.reservoir.shape[0] != size:
            if self.reservoir_filled != 0:
                raise RuntimeError(
                    "Cannot resize a non-empty historical reservoir."
                )
            self.reservoir = torch.empty(
                size,
                samples.shape[-1],
                dtype=torch.float32,
                device=samples.device,
            )
        elif self.reservoir.device != samples.device:
            self.reservoir = self.reservoir.to(samples.device)

        samples = samples.to(dtype=self.reservoir.dtype)

        n = samples.shape[0]
        filled = self.reservoir_filled
        seen = self.reservoir_seen

        if filled < size:
            n_fill = min(size - filled, n)
            self.reservoir[filled : filled + n_fill].copy_(samples[:n_fill])
            filled += n_fill
            seen += n_fill
            self.reservoir_filled = filled
            samples = samples[n_fill:]
            n = samples.shape[0]

        if n > 0:
            positions = (
                torch.arange(1, n + 1, device=samples.device, dtype=torch.float32)
                + seen
            )
            keep = torch.rand(n, device=samples.device) < (size / positions)
            keep_idx = keep.nonzero(as_tuple=False).view(-1)

            if keep_idx.numel() > 0:
                slots = torch.randint(
                    0,
                    size,
                    (keep_idx.numel(),),
                    device=samples.device,
                )
                self.reservoir[slots] = samples[keep_idx]

            seen += n

        self.reservoir_seen = seen

    def _expire_codes_spectokenizer(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return 0

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return 0

        filled = self.reservoir_filled
        if filled > 0:
            pool = self.reservoir[:filled]
        elif batch_samples is not None:
            pool = rearrange(batch_samples, "... d -> (...) d")
        else:
            return 0

        self.replace_(pool, mask=expired_codes)
        self._broadcast_codebook_state()

        return expired_codes.sum().item()


    def _reset_pending(self):
        self.pending_cluster_size.zero_()
        self.pending_embed_sum.zero_()
        self.pending_steps.zero_()
        if self._use_split:
            self._synchronize_pending_cpu_copies()
            self._pending_samples_cpu.clear()
            self._pending_indices_cpu.clear()

    def flush_ema_updates(self):
        if not self._use_delayed_ema:
            return
        if self.pending_steps.item() == 0:
            return
        self._apply_delayed_ema_update(self.pending_cluster_size, self.pending_embed_sum, None)
        self._reset_pending()

    def train(self, mode: bool = True):
        if not mode and self._use_delayed_ema:
            self.flush_ema_updates()
        return super().train(mode)

    def _build_sorted_pending_cpu(self):
        all_indices_cpu = torch.cat(self._pending_indices_cpu, dim=0).view(-1)
        all_samples_cpu = torch.cat(self._pending_samples_cpu, dim=0)

        order = torch.argsort(all_indices_cpu)
        sorted_indices_cpu = all_indices_cpu[order]
        sorted_samples_cpu = all_samples_cpu[order]

        counts_cpu = torch.bincount(
            sorted_indices_cpu.to(torch.long),
            minlength=self.codebook_size,
        )

        offsets_cpu = torch.empty(
            self.codebook_size + 1,
            dtype=torch.long,
            device="cpu",
        )
        offsets_cpu[0] = 0
        offsets_cpu[1:] = torch.cumsum(counts_cpu, dim=0)

        return sorted_samples_cpu, sorted_indices_cpu, counts_cpu, offsets_cpu
    

    def _get_sorted_samples_for_code_cpu(
        self,
        sorted_samples_cpu,
        offsets_cpu,
        code_idx: int,
    ):
        start = offsets_cpu[code_idx].item()
        end = offsets_cpu[code_idx + 1].item()

        if end <= start:
            return None

        return sorted_samples_cpu[start:end]

    
    def _rank_split_candidates_fast_cpu(
        self,
        sorted_samples_cpu,
        sorted_indices_cpu,
        counts_cpu,
        n_needed: int,
    ):
        if n_needed <= 0:
            return []

        decay_safe_split_size = (
            2.0
            * float(self.threshold_ema_dead_code)
            / max(float(self.decay), 1e-8)
        )
        frequency_floor = 2.0 * float(self.cluster_size.mean().item())
        min_size = max(decay_safe_split_size, frequency_floor)

        pending_counts = counts_cpu.to(
            device=self.cluster_size.device,
            dtype=self.cluster_size.dtype,
        )

        candidate_mask = self.cluster_size > 5
        candidate_indices = candidate_mask.nonzero(as_tuple=False).view(-1)

        if candidate_indices.numel() == 0:
            return []

        if self.split_candidate_strategy == "frequency":
            sorted_idx = torch.argsort(
                self.cluster_size[candidate_indices],
                descending=True,
            )
            return candidate_indices[sorted_idx[:n_needed]].tolist()

        embed_cpu = self.embed.detach().cpu().float()
        sorted_indices_long = sorted_indices_cpu.to(torch.long)
        samples_f32 = sorted_samples_cpu.float()
        centroids_f32 = embed_cpu[sorted_indices_long]
        dists = ((samples_f32 - centroids_f32) ** 2).sum(dim=-1)

        sum_dists_cpu = torch.zeros(
            self.codebook_size,
            dtype=torch.float32,
            device="cpu",
        )
        sum_dists_cpu.index_add_(0, sorted_indices_long, dists)
        var_all_cpu = sum_dists_cpu / counts_cpu.float().clamp_min(1.0)

        candidate_indices_cpu = candidate_indices.cpu()
        freq_scores = (
            self.cluster_size.detach().cpu()[candidate_indices_cpu].float()
        )
        var_scores = var_all_cpu[candidate_indices_cpu]

        if self.split_candidate_strategy == "frequency_variance":
            norm_freq = freq_scores / (freq_scores.max() + 1e-8)
            norm_var = var_scores / (var_scores.max() + 1e-8)
            combined = (
                self.split_frequency_weight * norm_freq
                + self.split_variance_weight * norm_var
            )
        elif self.split_candidate_strategy == "frequency_variance_product":
            combined = (
                freq_scores ** self.split_frequency_weight
                * var_scores ** self.split_variance_weight
            )
        else:
            raise ValueError(
                "Unsupported split_candidate_strategy: "
                f"{self.split_candidate_strategy}"
            )

        sorted_idx = torch.argsort(combined, descending=True)
        selected = candidate_indices_cpu[sorted_idx[:n_needed]]
        return selected.tolist()

    def _copy_to_pinned_cpu(self, tensor: torch.Tensor):
        out = torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device="cpu",
            pin_memory=True,
        )
        out.copy_(tensor.detach(), non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self._pending_copy_events.append(event)

        return out


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
        ema_update_every: int = 1,
        dead_code_strategy: str = "random",
        split_candidate_strategy: str = "frequency",
        split_frequency_weight: float = 1.0,
        split_variance_weight: float = 1.0,
        spectokenizer_bank_size: int = 32768,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code,
                                           ema_update_every=ema_update_every,
                                           dead_code_strategy=dead_code_strategy,
                                           split_candidate_strategy=split_candidate_strategy,
                                           split_frequency_weight=split_frequency_weight,
                                           split_variance_weight=split_variance_weight,
                                           spectokenizer_bank_size=spectokenizer_bank_size)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):

        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # warnings.warn('When using RVQ in training model, first check '
            #               'https://github.com/facebookresearch/encodec/issues/25 . '
            #               'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, ema_update_every: int = 1,
                 dead_code_strategy: str = "random", split_candidate_strategy: str = "frequency",
                 split_frequency_weight: float = 1.0, split_variance_weight: float = 1.0,
                 spectokenizer_bank_size: int = 32768, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(ema_update_every=ema_update_every,
                                dead_code_strategy=dead_code_strategy,
                                split_candidate_strategy=split_candidate_strategy,
                                split_frequency_weight=split_frequency_weight,
                                split_variance_weight=split_variance_weight,
                                spectokenizer_bank_size=spectokenizer_bank_size, **kwargs)
             for _ in range(num_quantizers)]
        )

    def forward(
        self,
        x,
        n_q: tp.Optional[int] = None,
    ):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = (
            len(self.layers)
            if n_q is None
            else int(n_q)
        )

        n_q = _sync_n_q_across_ranks(
            n_q,
            x.device,
        )

        if not 1 <= n_q <= len(self.layers):
            raise ValueError(
                f"Invalid n_q={n_q}; "
                f"expected 1..{len(self.layers)}"
            )

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(
            torch.stack,
            (all_losses, all_indices),
        )

        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class LanguageVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, ema_update_every: int = 1,
                 dead_code_strategy: str = "random", split_candidate_strategy: str = "frequency",
                 split_frequency_weight: float = 1.0, split_variance_weight: float = 1.0,
                 spectokenizer_bank_size: int = 32768, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(ema_update_every=ema_update_every,
                                dead_code_strategy=dead_code_strategy,
                                split_candidate_strategy=split_candidate_strategy,
                                split_frequency_weight=split_frequency_weight,
                                split_variance_weight=split_variance_weight,
                                spectokenizer_bank_size=spectokenizer_bank_size, **kwargs)
             for _ in range(num_quantizers)]
        )

    def forward(
        self,
        x,
        n_q: tp.Optional[int] = None,
    ):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = (
            len(self.layers)
            if n_q is None
            else int(n_q)
        )

        n_q = _sync_n_q_across_ranks(
            n_q,
            x.device,
        )

        for layer in self.layers[:n_q]:
            quantized_out, indices, loss = layer(residual)
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(
            torch.stack,
            (all_losses, all_indices),
        )

        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
