import torch


@torch.no_grad()
def encode_sequential(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> tuple:
    """
    Run a single forward pass over the full prompt, then generate.
    Returns (output_text, past_key_values, input_ids).
    The caller handles timing; this function just does the work.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past_kv = out.past_key_values

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_kv,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    output_text = tokenizer.decode(
        gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return output_text, past_kv, input_ids
