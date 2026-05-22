import os
import json
import torch

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file
from .division_model import Model1, Model2



#This function brings model weight's dict structure and split it into two dicts.
def split_llama_state_dict(full_state_dict, split_idx: int = 14):
    model1_state = {}
    model2_state = {}

    for key, value in full_state_dict.items():

        # embedding -> Model1
        if key == "model.embed_tokens.weight":
            model1_state["embed_tokens.weight"] = value

        # decoder layers
        elif key.startswith("model.layers."):
            parts = key.split(".")
            layer_idx = int(parts[2])

            rest = ".".join(parts[3:])

            if layer_idx < split_idx:
                # model.layers.0.xxx -> layers.0.xxx
                new_key = f"layers.{layer_idx}.{rest}"
                model1_state[new_key] = value
            else:
                # model.layers.14.xxx -> layers.0.xxx
                local_idx = layer_idx - split_idx
                new_key = f"layers.{local_idx}.{rest}"
                model2_state[new_key] = value

        # final norm -> Model2
        elif key == "model.norm.weight":
            model2_state["norm.weight"] = value

        # lm_head -> Model2
        elif key == "lm_head.weight":
            model2_state["lm_head.weight"] = value

    return model1_state, model2_state

def check_state_dict_match(target_model, source_state, name: str):
    target_state = target_model.state_dict()

    missing = []
    unexpected = []
    shape_mismatch = []

    for k in target_state.keys():
        if k not in source_state:
            missing.append(k)
        elif target_state[k].shape != source_state[k].shape:
            shape_mismatch.append((k, target_state[k].shape, source_state[k].shape))

    for k in source_state.keys():
        if k not in target_state:
            unexpected.append(k)

    print(f"\n[{name}] state_dict check")
    print(f"  missing: {len(missing)}")
    print(f"  unexpected: {len(unexpected)}")
    print(f"  shape_mismatch: {len(shape_mismatch)}")

    if missing:
        print("  missing examples:", missing[:10])
    if unexpected:
        print("  unexpected examples:", unexpected[:10])
    if shape_mismatch:
        print("  shape mismatch examples:", shape_mismatch[:5])

    if missing or unexpected or shape_mismatch:
        raise RuntimeError(f"{name} state_dict does not match.")
    
def verify_split_model(full_model, model1, model2, config, device="cpu"):
    full_model.eval()
    model1.eval()
    model2.eval()

    full_model.to(device)
    model1.to(device)
    model2.to(device)

    # seq_len=1로 검증하면 causal mask 차이 문제를 피할 수 있음
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 1),
        device=device,
    )

    with torch.no_grad():
        full_out = full_model(
            input_ids=input_ids,
            use_cache=True,
        )

        out1 = model1(
            input_ids=input_ids,
            use_cache=True,
        )

        out2 = model2(
            hidden_states=out1["hidden_states"],
            use_cache=True,
        )

    full_logits = full_out.logits
    split_logits = out2["logits"]

    max_diff = (full_logits - split_logits).abs().max().item()

    print("\n[verification]")
    print("  full logits shape :", tuple(full_logits.shape))
    print("  split logits shape:", tuple(split_logits.shape))
    print("  max abs diff      :", max_diff)

    if max_diff > 1e-4:
        raise RuntimeError(f"Split verification failed. max_diff={max_diff}")

    print(" split model output matches full model")

def download_model(
    repo_id: str = "unsloth/Llama-3.2-3B-Instruct",
    local_dir: str = "./models/Llama-3.2-3B-Instruct",
    output_dir: str = "./models/Llama-3.2-3B-Instruct-split",
    split_idx: int = 14,
    torch_dtype=torch.float32,
    device: str = "cpu",
):
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading {repo_id} -> {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.gguf", "*.bin"],
    )

    print("\nLoading full model...")

    config = AutoConfig.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)

    full_model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    print("\nCreating split models...")
    config = full_model.config
    model1 = Model1(config=config)
    model2 = Model2(config=config)

    print("\nSplitting state_dict...")

    full_state = full_model.state_dict()
    model1_state, model2_state = split_llama_state_dict(
        full_state,
        split_idx=split_idx,
    )

    check_state_dict_match(model1, model1_state, "Model1")
    check_state_dict_match(model2, model2_state, "Model2")

    print("\nLoading weights into split models...")

    model1.load_state_dict(model1_state, strict=True)
    model2.load_state_dict(model2_state, strict=True)

    verify_split_model(
        full_model=full_model,
        model1=model1,
        model2=model2,
        config=config,
        device=device,
    )

    print("\nSaving split checkpoint...")

    # torch.save(model1.state_dict(), os.path.join(output_dir, "model1_state.pt"))
    # torch.save(model2.state_dict(), os.path.join(output_dir, "model2_state.pt"))
    save_file(model1.state_dict(), os.path.join(output_dir, "model1.safetensors"))
    save_file(model2.state_dict(), os.path.join(output_dir, "model2.safetensors"))

    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "repo_id": repo_id,
        "local_dir": os.path.abspath(local_dir),
        "output_dir": os.path.abspath(output_dir),
        "split_idx": split_idx,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
    }

    with open(os.path.join(output_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n [results]Done")
    print(f"  model1: {os.path.join(output_dir, 'model1_state.pt')}")
    print(f"  model2: {os.path.join(output_dir, 'model2_state.pt')}")
    print(f"  config/tokenizer/meta saved to: {output_dir}")

    return model1, model2

















# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# import torch.nn as nn
# from transformers.models.llama.modeling_llama import LlamaForCausalLM

# from huggingface_hub import snapshot_download
# import os

# from .division_model import Model1, Model2

# # MODEL_PATH = "models/Llama-3.2-3B-Instruct"

# # ── config ────────────────────────────────────────────────────────────────────
# # Meta's official 3B instruct model (requires HF token + Meta license approval)
# # REPO_ID = "meta-llama/Llama-3.2-3B-Instruct"




# def download_model():
#     # Community fp16 snapshot — no token required
#     REPO_ID   = "bartowski/Llama-3.2-3B-Instruct-GGUF"   # GGUF weights (lightweight)
#     # Or use the full HF-format model:
#     REPO_ID   = "unsloth/Llama-3.2-3B-Instruct"           # full safetensors, no token
#     LOCAL_DIR = "./models/Llama-3.2-3B-Instruct"

#     os.makedirs(LOCAL_DIR, exist_ok=True)

#     print(f"Downloading {REPO_ID} → {LOCAL_DIR}")
#     print("    This may take a while (≈6 GB) …\n")

#     snapshot_download(
#         repo_id   = REPO_ID,
#         local_dir = LOCAL_DIR,
#         ignore_patterns=["*.gguf", "*.bin"],   # prefer safetensors
#     )

#     print("\n[✓] Download complete!")
#     print(f"    Saved to: {os.path.abspath(LOCAL_DIR)}")

#     model = AutoModelForCausalLM.from_pretrained(
#         LOCAL_DIR,
#         torch_dtype=torch.float32
#     )
#     print("______________________LLAMA3.2 MODEL________________________")
#     print(model)
#     # print("____________________________________________________________")

#     print("  __________________________________________________________")
#     print(" |                                                          | ")
#     print(" |           LLAMA3.2 MODEL PIPELINE SEPARATION             | ")
#     print(" |                                                          | ")
#     print("  __________________________________________________________")

#     config = AutoConfig.from_pretrained(LOCAL_DIR)
#     model1 = Model1(config=config)
#     model2 = Model2(config=config)

#     model

# # device = "cpu"
# # prompt = "KV cache가 뭐야?"

# # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# # model = AutoModelForCausalLM.from_pretrained(
# #     MODEL_PATH,
# #     torch_dtype=torch.float32,
# # ).to(device)

# # model.eval()