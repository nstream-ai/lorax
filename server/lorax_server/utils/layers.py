import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F

from lorax_server.adapters.types import LORA, MEDUSA
from lorax_server.utils.gptq.quant_linear import QuantLinear
from lorax_server.utils.sgmv import (
    has_sgmv,
    lora_a_sgmv_cutlass,
    lora_b_sgmv_cutlass,
    orient_for_rank,
)
from lorax_server.utils.state import is_warmup

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params, Params4bit

except ImportError:
    HAS_BITS_AND_BYTES = False

HAS_AWQ = True
try:
    from lorax_server.utils.awq.awq import AWQLinear
except ImportError:
    HAS_AWQ = False

HAS_EETQ = False
try:
    from EETQ import quant_weights, w8_a16_gemm

    HAS_EETQ = True
except ImportError:
    pass

HAS_HQQ = True
try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    class HQQLinearLayer(HQQLinear):
        @property
        def weight(self) -> torch.Tensor:
            return self.W_q

except ImportError:
    HAS_HQQ = False

HAS_EXLLAMA = True
if os.getenv("DISABLE_EXLLAMA") == "True":
    HAS_EXLLAMA = False
try:
    from lorax_server.utils.gptq.exllamav2 import QuantLinear as exllamav2QuantLinear
except ImportError:
    HAS_EXLLAMA = False

if TYPE_CHECKING:
    from lorax_server.adapters import AdapterBatchData
    from lorax_server.adapters.lora import BatchLoraWeights
    from lorax_server.adapters.medusa import BatchMedusaWeights


# Monkey patching
@classmethod
def load_layer_norm(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    bias = weights.get_tensor(f"{prefix}.bias")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = nn.Parameter(bias)
    return ln


@classmethod
def load_layer_norm_no_bias(cls, prefix, weights, eps):
    weight = weights.get_tensor(f"{prefix}.weight")
    with init_empty_weights():
        ln = cls(weight.shape, eps=eps)

    ln.weight = nn.Parameter(weight)
    ln.bias = None
    return ln


torch.nn.LayerNorm.load = load_layer_norm
torch.nn.LayerNorm.load_no_bias = load_layer_norm_no_bias


class FastLinear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


class FastConv1D(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
        self.nf = weight.shape[1]

    @classmethod
    def load(cls, config, prefix: str, weights):
        weight = weights.get_tensor(f"{prefix}.weight")
        bias = weights.get_tensor(f"{prefix}.bias")
        return cls(weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        size_out = input.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, input.view(-1, input.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class EETQLinear(nn.Module):
    """
    EETQLinear module applies quantized linear transformation to the input tensor.

    Args:
        weight (torch.Tensor): The weight tensor for the linear transformation.
        bias (torch.Tensor): The bias tensor for the linear transformation.

    Attributes:
        weight (torch.Tensor): The weight tensor for the linear transformation.
        scale (torch.Tensor): The scale tensor used for quantization.
        bias (torch.Tensor): The bias tensor for the linear transformation.

    """

    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        # Get the device where the weight tensor is currently stored.
        device = weight.device

        # Transpose the weight tensor and make a contiguous copy of it on the CPU.
        # The contiguous() function is used to ensure that the tensor is stored in a contiguous block of memory,
        # which can improve performance in some cases.
        weight_transposed = torch.t(weight)
        weight_contiguous = weight_transposed.contiguous()
        weight_cpu = weight_contiguous.cpu()

        # Quantize the weights. The quant_weights function is assumed to perform the quantization.
        # The weights are quantized to int8 format, and the quantization is not performed in place (False).
        weight_quantized, scale = quant_weights(weight_cpu, torch.int8, False)

        # Move the quantized weights and the scale back to the original device (GPU if available).
        # The cuda() function is used to move the tensors to the GPU.
        self.weight = weight_quantized.cuda(device)
        self.scale = scale.cuda(device)

        # If a bias is present, move it to the GPU as well. If not, set the bias to None.
        if bias is not None:
            self.bias = bias.cuda(device)
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the layer.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # The function w8_a16_gemm performs a matrix multiplication operation between the input and the weight of the layer.
        # The result is then scaled by a factor (self.scale).
        gemm_output = w8_a16_gemm(input, self.weight, self.scale)

        # If a bias is present (i.e., self.bias is not None), it is added to the output of the matrix multiplication.
        # If a bias is not present (i.e., self.bias is None), the output of the matrix multiplication is returned as is.
        if self.bias is not None:
            final_output = gemm_output + self.bias
        else:
            final_output = gemm_output

        # The final output is returned.
        return final_output


class Linear8bitLt(nn.Module):
    def __init__(
        self,
        weight,
        bias,
        has_fp16_weights=True,
        memory_efficient_backward=False,
        threshold=0.0,
        index=None,
    ):
        super().__init__()
        assert (
            not memory_efficient_backward
        ), "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
        self.state = bnb.MatmulLtState()
        self.index = index

        # Necessary for stacked layers
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        self.state.memory_efficient_backward = memory_efficient_backward
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True

        self.weight = Int8Params(
            weight.data,
            has_fp16_weights=has_fp16_weights,
            requires_grad=has_fp16_weights,
        )
        self.weight.cuda(weight.device)
        self.bias = bias

    def init_8bit_state(self):
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def forward(self, x: torch.Tensor):
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights:
            if self.state.CB is not None and self.state.CxB is not None:
                # we converted 8-bit row major to turing/ampere format in the first inference pass
                # we no longer need the row-major weight
                del self.state.CB
                self.weight.data = self.state.CxB
        return out


class Linear4bit(nn.Module):
    def __init__(self, weight, bias, quant_type):
        super().__init__()

        # Initialize weight with 4-bit quantization
        self.weight = Params4bit(weight.data, requires_grad=False, compress_statistics=True, quant_type=quant_type)
        self.weight.cuda(weight.device)

        # Initialize other attributes
        self.compute_dtype = None
        self.bias = bias

    def forward(self, x: torch.Tensor):
        # Ensure bias has the same dtype as input x
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        # Check if quantization state is initialized
        if getattr(self.weight, "quant_state", None) is None:
            print(
                "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first."
            )

        # Convert input to compute_dtype if specified
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        # Convert bias to compute_dtype if it exists
        bias = None if self.bias is None else self.bias.to(self.compute_dtype)

        # Perform 4-bit matrix multiplication
        out = bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state)

        # Convert output back to the input dtype
        out = out.to(inp_dtype)

        return out


# from float8_experimental.float8_linear_utils import (
#     swap_linear_with_float8_linear,
# )
# from float8_experimental.float8_dynamic_linear import Float8DynamicLinear

# from float8_experimental.float8_linear import Float8Linear


def fp8_quantize(weight, qdtype=torch.float8_e4m3fn):
    device = weight.device
    # weight, scale = quant_weights(weight, torch.int8, False)
    finfo = torch.finfo(qdtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / weight.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (weight * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(qdtype)
    scale = scale.float().reciprocal()
    return qweight, scale


class Fp8Linear(nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.dtype = weight.dtype
        self.qweight, self.scale = fp8_quantize(weight)

        self.bias = bias if bias is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qinput, scale = fp8_quantize(input)
        output, _ = torch._scaled_mm(
            qinput,
            self.qweight.t(),
            out_dtype=self.dtype,
            scale_a=scale,
            scale_b=self.scale,
            bias=self.bias,
        )
        return output


def get_linear(weight, bias, quantize, fan_in_fan_out=False):
    # https://huggingface.co/docs/peft/package_reference/tuners#peft.LoraConfig.fan_in_fan_out
    # Set to True if replacing a Conv1D layer with a Linear layer
    if fan_in_fan_out:
        weight = weight.T.contiguous()

    if True:
        # linear = FastLinear(weight, bias)
        # linear.in_features = weight.shape[1]
        # linear.out_features = weight.shape[0]
        # linear = Float8DynamicLinear.from_float(linear)

        linear = Fp8Linear(weight, bias)
    elif quantize is None:
        linear = FastLinear(weight, bias)
    elif quantize == "bitsandbytes":
        linear = Linear8bitLt(
            weight,
            bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        if bias is not None:
            linear.bias = nn.Parameter(bias)
    elif quantize == "bitsandbytes-nf4":
        linear = Linear4bit(
            weight,
            bias,
            quant_type="nf4",
        )
    elif quantize == "bitsandbytes-fp4":
        linear = Linear4bit(
            weight,
            bias,
            quant_type="fp4",
        )
    elif quantize == "eetq":
        if HAS_EETQ:
            linear = EETQLinear(weight, bias)
        else:
            raise ImportError("Please install EETQ from https://github.com/NetEase-FuXi/EETQ")
    elif quantize == "gptq":
        try:
            qweight, qzeros, scales, g_idx, bits, groupsize, use_exllama = weight
        except Exception:
            raise NotImplementedError("The passed weight is not `gptq` compatible, loader needs to be updated.")

        if use_exllama:
            linear = exllamav2QuantLinear(qweight, qzeros, scales, g_idx, bias, bits, groupsize)
        else:
            linear = QuantLinear(
                qweight,
                qzeros,
                scales,
                g_idx,
                bias,
                bits,
                groupsize,
            )
    elif quantize == "awq":
        try:
            qweight, qzeros, scales, _, bits, groupsize, _ = weight
        except Exception:
            raise NotImplementedError("The passed weight is not compatible with `awq`")
        linear = AWQLinear(
            w_bit=bits,
            group_size=groupsize,
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            bias=bias is not None,
        )
    elif "hqq-" in quantize:
        if quantize == "hqq-4bit":
            quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False)
        elif quantize == "hqq-3bit":
            quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_zero=True, quant_scale=False)
        elif quantize == "hqq-2bit":
            quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_zero=True, quant_scale=False)

        # init nn.linear from weight and bias
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        with torch.no_grad():
            layer.weight.data = weight
            if bias is not None:
                layer.bias.data = bias

        linear = HQQLinearLayer(layer, quant_config, del_orig=True)
    else:
        raise NotImplementedError(f"Quantization `{quantize}` is not implemented yet.")
    return linear


class SuperLayer(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear.forward(x)


class TensorParallelHead(SuperLayer):
    def __init__(self, linear, process_group, should_gather: bool):
        super().__init__(linear)
        self.process_group = process_group
        self.should_gather = should_gather

    @staticmethod
    def load(config, prefix: str, weights):
        if weights.process_group.size() > 1:
            try:
                weight = weights.get_sharded(f"{prefix}.weight", dim=0)
                should_gather = True
            except AssertionError:
                # If the vocab size is not divisible by number of shards
                # just load the entire thing.
                weight = weights.get_tensor(f"{prefix}.weight")
                should_gather = False
        else:
            weight = weights.get_tensor(f"{prefix}.weight")
            should_gather = False

        if config.quantize in ["gptq", "awq", "eetq"]:
            quantize = None
        else:
            quantize = config.quantize
        return TensorParallelHead(
            get_linear(weight, bias=None, quantize=quantize),
            process_group=weights.process_group,
            should_gather=should_gather,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        world_size = self.process_group.size()
        # Fast branch for single requests
        if self.should_gather and len(input.shape) == 2 and isinstance(self.linear, FastLinear) and input.shape[0] == 1:
            out_dim = self.linear.weight.shape[0]

            world_out = input.new_empty(1, out_dim * world_size)
            local_out = input.new_empty(1, out_dim)

            torch.mm(input, self.linear.weight.T, out=local_out)

            torch.distributed.all_gather_into_tensor(world_out, local_out, group=self.process_group)
            return world_out

        output = super().forward(input)
        if not self.should_gather:
            return output

        world_output = [torch.empty_like(output) for _ in range(world_size)]
        torch.distributed.all_gather(world_output, output, group=self.process_group)
        world_output = torch.cat(world_output, dim=-1)
        return world_output


class TensorParallelColumnLinear(SuperLayer):
    @classmethod
    def load_qkv(cls, config, prefix: str, weights, bias: bool, fan_in_fan_out=False):
        """Specific method when the QKV was joined after the fact"""
        weight = weights.get_weights_col_packed_qkv(prefix, quantize=config.quantize)
        if bias:
            raise NotImplementedError("packed_qkv only implemented for baichuan")
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize, fan_in_fan_out=fan_in_fan_out)
        return cls(linear)

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool, fan_in_fan_out: bool = False):
        return cls.load_multi(config, [prefix], weights, bias, dim=0, fan_in_fan_out=fan_in_fan_out)

    @classmethod
    def load_multi(
        cls,
        config,
        prefixes: List[Union[str, Tuple]],
        weights,
        bias: bool,
        dim: int,
        fan_in_fan_out=False,
    ):
        weight = weights.get_multi_weights_col(prefixes, quantize=config.quantize, dim=dim)

        if bias:
            b = weights.get_sharded_list("bias", prefixes, dim=0)
            bias = torch.cat(b, dim=dim)
        else:
            bias = None
        linear = get_linear(weight, bias, config.quantize, fan_in_fan_out=fan_in_fan_out)
        return cls(linear)


class LoraLinear(nn.Module):
    def __init__(self, base_layer, layer_id, process_group):
        super().__init__()
        self.base_layer = base_layer
        self.layer_id = layer_id
        self.process_group = process_group

    def forward_layer_type(
        self,
        result: torch.Tensor,
        input: torch.Tensor,
        adapter_data: "AdapterBatchData",
        layer_type: str,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        data = adapter_data.data.get(layer_type)
        data: Optional["BatchLoraWeights"] = data.get(LORA) if data is not None else None

        if has_sgmv() and data is not None and data.can_vectorize(self.process_group):
            if end_idx - start_idx != result.shape[1]:
                proj = torch.zeros_like(result[:, start_idx:end_idx])
            else:
                proj = result

            for r, rank_segments in data.rank_data.items():
                lora_a_ptr = rank_segments.lora_a_ptr
                lora_b_ptr = rank_segments.lora_b_ptr
                if lora_a_ptr is not None and lora_b_ptr is not None:
                    v = lora_a_sgmv_cutlass(
                        input,
                        rank_segments.tmp_shrink,
                        lora_a_ptr,
                        rank_segments.segment_starts,
                        rank_segments.segment_ends,
                        self.layer_id,
                        r,
                    )

                    if self.process_group.size() > 1:
                        v = self.collect_lora_a(v)

                    lora_b_sgmv_cutlass(
                        proj,
                        v,
                        rank_segments.tmp_expand,
                        lora_b_ptr,
                        rank_segments.segment_starts,
                        rank_segments.segment_ends,
                        self.layer_id,
                    )

            if end_idx - start_idx != result.shape[1]:
                result[:, start_idx:end_idx] += proj
        else:
            for adapter_index in adapter_data.meta.adapter_set:
                if data is not None and data.has_adapter(adapter_index):
                    adapter_mask = (adapter_data.meta.adapter_indices == adapter_index).to(input.dtype).view(-1, 1)
                    layer_result = self.forward_lora(input, data, adapter_index, adapter_mask)
                    result[:, start_idx:end_idx] += layer_result

        return result

    def forward_lora(
        self,
        input: torch.Tensor,
        data: "BatchLoraWeights",
        adapter_index: int,
        adapter_mask: torch.Tensor,
    ) -> torch.Tensor:
        lora_a = data.lora_a[adapter_index][self.layer_id, :, :]
        lora_b = data.lora_b[adapter_index][self.layer_id, :, :]

        lora_a = orient_for_rank(lora_a, lora_b.size(0))

        a_out = input @ lora_a
        if self.process_group.size() > 1:
            a_out = self.collect_lora_a(a_out)

        result = (a_out @ lora_b) * adapter_mask
        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implemented in subclasses")


class TensorParallelMultiAdapterLinear(LoraLinear):
    def __init__(self, base_layer, layer_id, layer_names, sizes, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_names = layer_names
        self.sizes = sizes

    @classmethod
    def load(cls, base_layer, layer_id, layer_names, sizes, process_group):
        return TensorParallelMultiAdapterLinear(base_layer, layer_id, layer_names, sizes, process_group)

    def forward(self, input: torch.Tensor, adapter_data: "AdapterBatchData") -> torch.Tensor:
        result = self.base_layer(input)

        # handle models like Bloom that have inputs of shape
        # (batch_size, sequence_length, hidden_size)
        # we need to reshape them to (batch_size * sequence_length, hidden_size)
        # for the LoRA computation, then reshape back
        prev_shape = result.shape
        is_3d = len(input.shape) >= 3
        if is_3d:
            input = input.reshape(-1, input.shape[-1])
            result = result.reshape(-1, result.shape[-1])

        offset = 0
        for i, layer_name in enumerate(self.layer_names):
            start_idx = offset // self.process_group.size()

            if self.sizes is not None:
                offset += self.sizes[i]
                end_idx = offset // self.process_group.size()
            else:
                end_idx = result.shape[1]

            result = self.forward_layer_type(result, input, adapter_data, layer_name, start_idx, end_idx)

        if is_3d:
            result = result.reshape(prev_shape)

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded column-wise.
        # We use an all-gather between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-gather for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        gathered_tensors = [torch.empty_like(a_out) for _ in range(self.process_group.size())]
        torch.distributed.all_gather(gathered_tensors, a_out)
        return torch.cat(gathered_tensors, dim=1)


class TensorParallelAdapterRowLinear(LoraLinear):
    def __init__(self, base_layer, layer_id, layer_name, process_group):
        super().__init__(base_layer, layer_id, process_group)
        self.layer_name = layer_name

    @classmethod
    def load(cls, base_layer, layer_id, layer_name, process_group):
        return cls(base_layer, layer_id, layer_name, process_group)

    def forward(self, input: torch.Tensor, adapter_data: "AdapterBatchData") -> torch.Tensor:
        result = self.base_layer(input)

        # Fused all-gather + all-reduce from S-LoRA paper: https://arxiv.org/abs/2311.03285
        stride = result.shape[-1] // self.process_group.size()
        start_idx = self.process_group.rank() * stride
        end_idx = (self.process_group.rank() + 1) * stride

        self.forward_layer_type(result, input, adapter_data, self.layer_name, start_idx, end_idx)

        return result

    def collect_lora_a(self, a_out: torch.Tensor) -> torch.Tensor:
        # Tensor parallel implementation of X @ A@B, where A and B are sharded row-wise.
        # We use an all-reduce between X@A and (X@A)@B to ensure alignment across ranks.
        #
        # TODO(travis): this is not very efficient as we do an all-reduce for every adapter,
        #   instead we could pre-allocate a (B, a, r) tensor for all adapters with the same
        #   rank, compute `a_out` on each, and then slice them into the buffer as shown here:
        #   https://discuss.pytorch.org/t/concatenate-tensors-without-memory-copying/34609
        torch.distributed.all_reduce(a_out, group=self.process_group)
        return a_out


class MultiAdapterHead(TensorParallelAdapterRowLinear):
    def forward(
        self, input: torch.Tensor, adapter_data: "AdapterBatchData"
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        result = super().forward(input, adapter_data)

        # Medusa
        data = adapter_data.data.get(self.layer_name)
        data: Optional["BatchMedusaWeights"] = data.get(MEDUSA) if data is not None else None

        speculative_logits = None
        if data is not None and data.default_medusa is not None:
            speculative_logits = data.default_medusa.model(input)

            # TODO(travis): support multiple medusa adapters with masking:
            # for adapter_index in adapter_data.meta.adapter_set:
            #     if data.has_adapter(adapter_index):
            #         adapter_mask = (adapter_data.meta.adapter_indices == adapter_index).to(input.dtype).view(-1, 1)
            #         speculative_logits = data.adapter_to_medusa[adapter_index].model(input)
            #         ...

        return result, speculative_logits


class TensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group, all_reduce: bool = True):
        super().__init__(linear)
        self.process_group = process_group
        self.all_reduce = all_reduce

    @classmethod
    def load(
        cls,
        config,
        prefix: str,
        weights,
        bias: bool,
        fan_in_fan_out: bool = False,
        all_reduce: bool = True,
    ):
        weight = weights.get_multi_weights_row(prefix, quantize=config.quantize)

        if bias and weights.process_group.rank() == 0:
            # Rank is only on the first rank process
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(
            get_linear(weight, bias, config.quantize, fan_in_fan_out=fan_in_fan_out),
            process_group=weights.process_group,
            all_reduce=all_reduce,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1 and self.all_reduce:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class TensorParallelEmbedding(nn.Module):
    def __init__(self, prefix: str, weights, reduce=True):
        super().__init__()
        weight = weights.get_partial_sharded(f"{prefix}.weight", dim=0)
        num_embeddings = weights.get_shape(f"{prefix}.weight")[0]

        process_group = weights.process_group

        world_size = process_group.size()
        rank = process_group.rank()

        block_size = num_embeddings // world_size
        self.min_id = rank * block_size
        self.max_id = min(num_embeddings, (rank + 1) * block_size)
        self.null_idx = block_size
        self.process_group = weights.process_group
        self.reduce = reduce

        """Additional 0 entry used for masking"""
        self.weight = nn.Parameter(F.pad(weight, (0, 0, 0, 1)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # default all out of bounds values to `self.null_idx` that will then be mapped to 0
        # translate for [0, self.max_id - self.min_id[
        input = torch.where(
            (self.min_id > input) | (input >= self.max_id),
            self.null_idx,
            input - self.min_id,
        )
        out = torch.nn.functional.embedding(input, self.weight)
        if self.reduce and self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


try:
    import dropout_layer_norm

    class FastLayerNorm(nn.LayerNorm):
        def forward(self, hidden_states, residual=None):
            if hidden_states.shape[-1] > 8192:
                if residual is not None:
                    hidden_states += residual
                residual = hidden_states

                return super(FastLayerNorm, self).forward(hidden_states), residual
            else:
                (
                    normed_hidden_states,
                    residual,
                    *rest,
                ) = dropout_layer_norm.dropout_add_ln_fwd(
                    hidden_states,
                    residual,
                    self.weight,
                    self.bias,
                    None,
                    None,
                    None,
                    None,
                    0.0,
                    self.eps,
                    1.0,
                    0,
                    None,
                    False,
                    False,
                )
                if residual is None:
                    residual = hidden_states

                return normed_hidden_states, residual

except ImportError:
    pass


try:
    import rotary_emb
    from flash_attn.layers.rotary import RotaryEmbedding  # noqa: F401

    def _create_inv_freq(dim, base, device):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        return inv_freq

    def _get_rope_config(config):
        if os.getenv("ROPE_SCALING", None) is not None:
            rope_scaling = {
                "type": os.environ["ROPE_SCALING"],
                "factor": float(os.environ["ROPE_FACTOR"]),
            }
            return rope_scaling
        return getattr(config, "rope_scaling", None)

    class PositionRotaryEmbedding(nn.Module):
        def __init__(self, inv_freq, scaling_factor, max_position_embeddings, device, dtype):
            super().__init__()
            self.inv_freq = inv_freq
            self._seq_len_cached = 0
            self._cos_cached = None
            self._sin_cached = None
            self._cos_k_cached = None
            self._sin_k_cached = None
            self.scaling_factor = scaling_factor
            self.dynamic_args = None
            self._update_cos_sin_cache(dtype, device, max_position_embeddings)

        @classmethod
        def static(cls, config, dim, base, device, dtype):
            inv_freq = _create_inv_freq(dim, base, device)
            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                rope_scaling = rope_scaling.copy()
                scaling_factor = rope_scaling["factor"]
                rope_type = rope_scaling.pop("type")
                if rope_type == "linear":
                    pass
                elif rope_type == "dynamic":
                    return DynamicPositionRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=config.max_position_embeddings,
                        base=base,
                        device=inv_freq.device,
                        dtype=dtype,
                        scaling_factor=scaling_factor,
                    )
                elif rope_type == "yarn":
                    return YarnPositionRotaryEmbedding(
                        dim=dim,
                        max_position_embeddings=config.max_position_embeddings,
                        base=base,
                        device=inv_freq.device,
                        dtype=dtype,
                        **rope_scaling,
                    )
                else:
                    raise NotImplementedError(f"rope scaling type {rope_type} is not implemented or invalid")
            return cls(inv_freq, scaling_factor, config.max_position_embeddings, device, dtype)

        @classmethod
        def load(cls, config, prefix, weights):
            # XXX: Always load this in float32 !
            dtype = weights.dtype
            weights.dtype = torch.float32
            inv_freq = weights.get_tensor(f"{prefix}.inv_freq")
            weights.dtype = dtype

            scaling_factor = None
            rope_scaling = _get_rope_config(config)
            if rope_scaling is not None:
                rope_scaling = rope_scaling.copy()
                scaling_factor = rope_scaling["factor"]
                rope_type = rope_scaling.pop("type")
                if rope_type == "linear":
                    pass
                elif rope_type == "dynamic":
                    return DynamicPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=config.max_position_embeddings,
                        base=10000.0,
                        device=inv_freq.device,
                        dtype=dtype,
                        scaling_factor=scaling_factor,
                    )
                elif rope_type == "yarn":
                    return YarnPositionRotaryEmbedding(
                        dim=2 * inv_freq.shape[0],
                        max_position_embeddings=config.max_position_embeddings,
                        base=10000.0,
                        device=inv_freq.device,
                        dtype=dtype,
                        **rope_scaling,
                    )
                else:
                    raise NotImplementedError(f"rope scaling type {rope_type} is not implemented or invalid")
            return cls(inv_freq, scaling_factor)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                if self.scaling_factor is not None:
                    t /= self.scaling_factor
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

        def get_cos_sin(self, position_ids: torch.Tensor, max_s: int, dtype: torch.dtype):
            """
            Return cos and sin for the asked position ids
            """

            # When using dynamic position embeddings, the max sequence length might exceed
            # the max position embeddings of the base model, so we need to update our
            # cache during warmup.
            # This should never result in a change after warmup, otherwise we break
            # cuda graphs.
            if is_warmup():
                self._update_cos_sin_cache(dtype, position_ids.device, max_s)

            cos = torch.index_select(self._cos_cached, 0, position_ids)
            sin = torch.index_select(self._sin_cached, 0, position_ids)
            return cos.unsqueeze(1), sin.unsqueeze(1)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
            rotary_dim = cos.shape[-1]
            x1 = x[..., :rotary_dim]
            x2 = x[..., rotary_dim : 2 * rotary_dim]

            rotary_emb.apply_rotary(x1, x2, cos, sin, x1, x2, False)
            return x

    class DynamicPositionRotaryEmbedding(PositionRotaryEmbedding):
        def __init__(self, dim, max_position_embeddings, base, device, dtype, scaling_factor):
            inv_freq = _create_inv_freq(dim, base, device)
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            super().__init__(inv_freq, scaling_factor, max_position_embeddings, device, dtype)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                if seqlen > self.max_position_embeddings:
                    newbase = self.base * (
                        (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
                    ) ** (self.dim / (self.dim - 2))
                    self.inv_freq = _create_inv_freq(self.dim, newbase, self.inv_freq.device)
                self._seq_len_cached = seqlen
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                # Don't do einsum, it converts fp32 to fp16
                # freqs = torch.einsum("i,j->ij", t, self.inv_freq)

                freqs = torch.outer(t, self.inv_freq.to(device=t.device))
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)

    class YarnPositionRotaryEmbedding(PositionRotaryEmbedding):
        """https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py"""

        def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            factor=1,
            original_max_position_embeddings=2048,
            extrapolation_factor=1,
            attn_factor=1,
            beta_fast=32,
            beta_slow=1,
            finetuned=True,
            device=None,
            dtype=None,
        ):
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.original_max_position_embeddings = original_max_position_embeddings
            self.extrapolation_factor = extrapolation_factor
            self.attn_factor = attn_factor
            self.beta_fast = beta_fast
            self.beta_slow = beta_slow
            self.finetuned = finetuned

            self.yarn(device, factor)
            super().__init__(_create_inv_freq(dim, base, device), factor, max_position_embeddings, device, dtype)

        def _update_cos_sin_cache(self, dtype, device, seqlen):
            if seqlen > self._seq_len_cached or self._cos_cached.device != device or self._cos_cached.dtype != dtype:
                self._seq_len_cached = seqlen

                t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq.to(device=t.device))

                self._cos_cached = (torch.cos(freqs) * self.mscale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * self.mscale).to(dtype)

        def yarn(self, device, scaling_factor):
            pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

            low, high = find_correction_range(
                self.beta_fast,
                self.beta_slow,
                self.dim,
                self.base,
                self.original_max_position_embeddings,
            )
            inv_freq_mask = (
                1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
            ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

            self.inv_freq = inv_freq
            self.mscale = float(
                get_mscale(scaling_factor) * self.attn_factor
            )  # Get n-d magnitude scaling corrected for interpolation

    # Inverse dim formula to find dim based on number of rotations
    def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    def linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def get_mscale(scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

except ImportError:
    pass
