import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

MODEL_CACHE = {}
TOKENIZER_CACHE = {}
MODEL_LOCK = threading.Lock()

DEFAULT_MODEL = "distilgpt2"


def load_model(checkpoint_or_path: str = DEFAULT_MODEL, device: str = "cpu"):
    with MODEL_LOCK:
        if checkpoint_or_path not in MODEL_CACHE:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_or_path)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_or_path)
            model.to(device)
            MODEL_CACHE[checkpoint_or_path] = model
            TOKENIZER_CACHE[checkpoint_or_path] = tokenizer
    return MODEL_CACHE[checkpoint_or_path], TOKENIZER_CACHE[checkpoint_or_path]


def generate_response(
    input_text: str,
    model_tag: str = "stable",
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    checkpoint = get_checkpoint_for_tag(model_tag)
    model, tokenizer = load_model(checkpoint)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if output.startswith(input_text):
        return output[len(input_text):].strip()
    return output


def get_checkpoint_for_tag(tag: str) -> str:
    # For demo: map 'stable' and 'vnext' to distilgpt2, can be extended
    if tag in ("stable", "vnext"):
        return DEFAULT_MODEL
    return DEFAULT_MODEL
