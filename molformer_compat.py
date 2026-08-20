from __future__ import annotations

import sys
import types


def enable_molformer_compat() -> None:
    """Patch transformer API gaps needed by the IBM MoLFormer remote code.

    The local environment uses a newer Transformers build than the official
    MoLFormer code expects. These shims are minimal and only cover the APIs
    exercised by inference/fine-tuning in this experiment.
    """

    import torch

    if "transformers.onnx" not in sys.modules:
        mod = types.ModuleType("transformers.onnx")

        class OnnxConfig:  # pragma: no cover - compatibility shim
            pass

        mod.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = mod

    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(
            heads: set[int] | list[int],
            n_heads: int,
            head_size: int,
            already_pruned_heads: set[int],
        ):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                pruned_before = sum(1 for item in already_pruned_heads if item < head)
                mask[head - pruned_before] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
            if head_mask is None:
                return [None] * num_hidden_layers
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=self.dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask
