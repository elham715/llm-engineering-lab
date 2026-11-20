"""
Byte-Level Byte-Pair Encoding (BPE) Tokenizer from Scratch
Reference: Sennrich et al. (ACL 2016); Radford et al. (GPT-2, 2019 Section 2.2)

Key Architecture:
- Base vocabulary: 256 bytes (0..255), zero <unk> tokens across any language/symbol.
- Greedy pair counting and non-overlapping left-to-right merging.
- Lowest-merge-rank encoding preserving hierarchical dependency trees.
- Lossless round-trip byte decoding with UTF-8 replacement fallback.
"""

from __future__ import annotations
import json
from typing import Dict, List, Tuple


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Count frequency of every adjacent pair in an integer token sequence."""
    counts: Dict[Tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Replace every non-overlapping occurrence of `pair` with `new_id`.
    Crucial: advances index by 2 when a pair is matched to prevent double-counting.
    """
    out: List[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # Consumed both tokens in the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """
    Pure Python Byte-Level BPE Tokenizer.
    Base vocabulary: 0..255 (raw UTF-8 bytes).
    Merges: (token_a, token_b) -> new_token_id (256, 257, ...)
    """

    def __init__(self) -> None:
        # (int, int) -> int (new token id)
        self.merges: Dict[Tuple[int, int], int] = {}
        # int -> bytes representation
        self.vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Train BPE merge rules from raw text up to the target `vocab_size`.
        Base vocabulary occupies indices 0..255.
        Number of merges = vocab_size - 256.
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 for byte-level base."
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"Stopping early at step {i}: no more adjacent pairs to merge.")
                break

            # Find the most frequent adjacent pair
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i

            # Apply merge to training corpus
            ids = merge(ids, top_pair, new_id)

            # Record merge rule and vocabulary byte sequence
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose and (i < 5 or (i + 1) % 50 == 0 or i == num_merges - 1):
                print(
                    f"Merge {i + 1}/{num_merges}: pair {top_pair} -> {new_id} "
                    f"({self.vocab[new_id]!r}) | count: {stats[top_pair]}"
                )

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into token IDs using learned merge rules.
        Invariant: Always merges the lowest-merge-rank candidate pair first.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the adjacent pair present in `ids` that has the lowest merge ID (earliest learned)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # No more candidate pairs can be merged
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a human-readable UTF-8 string.
        Uses errors='replace' to prevent unhandled crashes on partial multi-byte streams.
        """
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        """Save merges and vocabulary to JSON."""
        # Convert tuple keys to string "p0,p1" for JSON serialization
        serializable_merges = {f"{p[0]},{p[1]}": mid for p, mid in self.merges.items()}
        data = {
            "merges": serializable_merges,
            "vocab_size": len(self.vocab),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load tokenizer from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # Reconstruct merges sorted by newly minted ID to preserve hierarchy
        sorted_items = sorted(data["merges"].items(), key=lambda kv: kv[1])
        for pair_str, new_id in sorted_items:
            p0_s, p1_s = pair_str.split(",")
            pair = (int(p0_s), int(p1_s))
            self.merges[pair] = int(new_id)
            self.vocab[int(new_id)] = self.vocab[pair[0]] + self.vocab[pair[1]]
