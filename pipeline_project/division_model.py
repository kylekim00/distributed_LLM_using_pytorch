import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


def choose_attention_backend(device: str = "cpu") -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except Exception:
            pass

    try:
        test_device = torch.device(device)
        q = torch.randn(1, 2, 4, 8, device=test_device)
        k = torch.randn(1, 2, 4, 8, device=test_device)
        v = torch.randn(1, 2, 4, 8, device=test_device)

        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return "sdpa"
    except Exception:
        pass
    return "eager"

class Model1(nn.Module):
    def __init__(self, config, use_cache: bool = False, device:str="cpu"):
        super().__init__()

        self.use_cache = use_cache
        self.config = config
        # self.config._attn_implementation = choose_attention_backend(device)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(14)
        ])
    def check_attn_backend(self):
        self.config._attn_implementation = choose_attention_backend(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ):
        if use_cache is None:
            use_cache = self.use_cache

        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        hidden_states = self.embed_tokens(input_ids)

        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        elif attention_mask is not None:
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs

        return {
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
        }

class Model2(nn.Module):
    def __init__(self, config, use_cache: bool = False, device:str='cpu'):
        super().__init__()

        self.use_cache = use_cache
        self.config = config
        # self.config._attn_implementation = choose_attention_backend(device)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(14, config.num_hidden_layers)
        ])

        self.norm = LlamaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
    def check_attn_backend(self):
        self.config._attn_implementation = choose_attention_backend(self.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ):
        if use_cache is None:
            use_cache = self.use_cache

        device = hidden_states.device
        batch_size, seq_len, _ = hidden_states.shape

        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        elif attention_mask is not None:
            position_ids = attention_mask.cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        else:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "past_key_values": past_key_values,
        }