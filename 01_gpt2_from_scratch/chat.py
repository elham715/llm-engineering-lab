"""
Interactive Few-Shot Assistant Chat with custom GPT-2 (124M).
Human asks a question -> GPT-2 answers cleanly as Assistant -> Human asks next question!
"""

import sys
import torch
import torch.nn.functional as F
from model import GPT2Model
from generate import load_tokenizer, pick_device

# Few-shot primer: teaches base GPT-2 what an "Assistant" behaves like!
SYSTEM_PROMPT = (
    "The following is an interview with an intelligent, concise AI assistant.\n\n"
    "Human: What is the capital of France?\n"
    "Assistant: The capital of France is Paris.\n\n"
    "Human: Who was Alan Turing?\n"
    "Assistant: Alan Turing was a British mathematician and pioneer of computer science.\n\n"
)


def assistant_chat():
    dev = pick_device("auto")
    encode, decode = load_tokenizer()

    print("\n" + "="*58)
    print(f" Loading custom GPT-2 (124M) Assistant onto {dev}...")
    print("="*58)
    model = GPT2Model.from_pretrained("gpt2").to(dev)
    model.eval()
    print("Assistant is ready! Ask it any question.")
    print("Type 'exit' or press Ctrl+C to stop.\n" + "-"*58)

    history = SYSTEM_PROMPT

    while True:
        try:
            user_msg = input("\nHuman: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Append human turn
        prompt = history + f"Human: {user_msg}\nAssistant:"
        input_ids = encode(prompt)
        if len(input_ids) > 850:
            # Keep system prompt + recent dialogue
            input_ids = encode(SYSTEM_PROMPT) + input_ids[-600:]

        idx = torch.tensor([input_ids], dtype=torch.long, device=dev)

        sys.stdout.write("Assistant: ")
        sys.stdout.flush()

        reply_tokens = []
        with torch.no_grad():
            for _ in range(60):  # max 60 tokens per response
                logits, _ = model(idx[:, -1024:])
                logits = logits[:, -1, :] / 0.75  # focused temperature
                v, _ = torch.topk(logits, 40)
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, idx_next), dim=1)
                tok_id = int(idx_next.item())
                tok_str = decode([tok_id])

                # STOP CONDITION: Cut immediately if it tries to start a new turn or newline
                if "\n" in tok_str or "Human:" in tok_str or "User:" in tok_str:
                    break

                reply_tokens.append(tok_id)
                sys.stdout.write(tok_str)
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Save this clean turn to history
        clean_reply = decode(reply_tokens).strip()
        history += f"Human: {user_msg}\nAssistant: {clean_reply}\n\n"


if __name__ == "__main__":
    assistant_chat()
