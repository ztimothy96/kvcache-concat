import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SUPPORTED = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}


def load_model(name: str, device: str = "cuda", dtype=torch.bfloat16):
    hf_id = SUPPORTED[name]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=dtype,
        device_map=device,
        attn_implementation=
        "eager",  # flash/sdpa fuse kernel and hide KV tensors
    )
    model.eval()
    return model, tokenizer
