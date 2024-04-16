from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed
from opentelemetry import trace
from transformers.models.llama import LlamaTokenizerFast

from lorax_server.models import FlashCausalLM
from lorax_server.models.types import FlashBatch
from lorax_server.models.custom_modeling.flash_mistral_modeling import (
    FlashMistralForCausalLM,
    FlashMistralForEmbedding,
    MistralConfig,
)
from lorax_server.utils import (
    Weights,
    initialize_torch_distributed,
    weight_files,
)
from lorax_server.utils.lora import (
    DOWN_PROJ,
    GATE_PROJ,
    K_PROJ,
    LM_HEAD,
    O_PROJ,
    Q_PROJ,
    UP_PROJ,
    V_PROJ,
)

tracer = trace.get_tracer(__name__)

ADAPTER_LAYERS = [Q_PROJ, K_PROJ, V_PROJ, O_PROJ, GATE_PROJ, UP_PROJ, DOWN_PROJ, LM_HEAD]
ROW_PARALLEL = {O_PROJ, DOWN_PROJ, LM_HEAD}


class FlashMistral(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        adapter_source: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        compile: bool = False,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        embed_mode: bool = False,
    ):
        self.embed_mode = embed_mode
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashLlama is only available on GPU")

        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        config = MistralConfig.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
        config.quantize = quantize

        torch.distributed.barrier(group=self.process_group)

        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
        )
        weights._set_config(model_id, config)

        if self.embed_mode:
            model = FlashMistralForEmbedding(config, weights)
        else:
            model = FlashMistralForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashMistral, self).__init__(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            num_layers=len(model.model.layers),
            num_kv_heads=model.model.num_key_value_heads,
            head_size=model.model.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
            sliding_window=config.sliding_window,
            compile=compile,
            adapter_id=adapter_id,
            adapter_source=adapter_source,
        )

    @property
    def supports_adapter_loading(self) -> bool:
        return True

    def adapter_target_to_layer(self) -> Dict[str, Tuple[str, torch.Tensor]]:
        layer_weights = {}

        prefix = "model.layers"
        for i, layer in enumerate(self.model.model.layers):
            layer_weights[(i, Q_PROJ)] = (
                f"{prefix}.{i}.self_attn.q_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, K_PROJ)] = (
                f"{prefix}.{i}.self_attn.k_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, V_PROJ)] = (
                f"{prefix}.{i}.self_attn.v_proj",
                layer.self_attn.query_key_value,
            )
            layer_weights[(i, O_PROJ)] = (f"{prefix}.{i}.self_attn.o_proj", layer.self_attn.o_proj)

            layer_weights[(i, GATE_PROJ)] = (f"{prefix}.{i}.mlp.gate_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, UP_PROJ)] = (f"{prefix}.{i}.mlp.up_proj", layer.mlp.gate_up_proj)
            layer_weights[(i, DOWN_PROJ)] = (f"{prefix}.{i}.mlp.down_proj", layer.mlp.down_proj)
        
        if self.embed_mode:
            return layer_weights

        layer_weights[(0, LM_HEAD)] = ("lm_head", self.model.lm_head)
        return layer_weights

    @property
    def adapter_layers(self) -> List[str]:
        return ADAPTER_LAYERS

    def get_num_layers_for_type(self, layer_type: str) -> int:
        return 1 if layer_type == LM_HEAD else len(self.model.model.layers)

    def is_row_parallel(self, layer_type: str) -> bool:
        return layer_type in ROW_PARALLEL

    def tokenize_to_batch(self, inputs) -> FlashBatch:
        tokens = self.tokenizer(inputs, return_token_type_ids=True)
        num_tokens = len(tokens["input_ids"])
        position_ids = range(num_tokens)
        return FlashBatch(
            input_ids=torch.tensor(tokens["input_ids"], dtype=torch.int32, device=self.device),
            token_type_ids=torch.tensor(tokens["token_type_ids"], dtype=torch.int32, device=self.device),
            position_ids=torch.tensor(position_ids, dtype=torch.int32, device=self.device),
            cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device),
            cu_seqlen_prefill=None,
            max_s=num_tokens,
            size=1,
        )
