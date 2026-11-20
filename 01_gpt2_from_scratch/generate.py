"""
End-to-End Autoregressive GPT-2 Generation Script with Live Streaming.
Chapter 04: Loads official OpenAI weights into custom GPT2Model and streams tokens.

Usage:
    python generate.py --prompt "Hello, I am a language model," --max-new-tokens 40
"""

import argparse
import sys
import torch

from model import GPT2Model


def load_tokenizer():
    """Prefer tiktoken for fast BPE encoding/decoding; fall back to HuggingFace GPT2Tokenizer."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.decode
    except ImportError:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return tok.encode, tok.decode


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stream_generate(
    prompt: str = "Hello, I am a language model,",
    model_type: str = "gpt2",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "auto",
) -> str:
    """Load pretrained GPT-2, encode prompt, and stream generated tokens live to stdout."""
    dev = pick_device(device)
    encode, decode = load_tokenizer()

    print(f"Loading '{model_type}' onto {dev}...")
    model = GPT2Model.from_pretrained(model_type)
    model.to(dev)
    model.eval()
    print(f"Loaded {model.count_parameters():,} unique parameters.\n")

    prompt_ids = encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    sys.stdout.write(prompt)
    sys.stdout.flush()

    def on_token(tok_id: int) -> None:
        sys.stdout.write(decode([tok_id]))
        sys.stdout.flush()

    out_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        on_token=on_token,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()
    return decode(out_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream text from custom GPT-2 (124M).")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model,")
    parser.add_argument("--model-type", type=str, default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    stream_generate(
        prompt=args.prompt,
        model_type=args.model_type,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()

