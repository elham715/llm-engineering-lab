"""
Byte-Level BPE Tokenizer from Scratch
Radford et al. (GPT-2, 2019) / Sennrich et al. (ACL 2016)
"""

def get_stats(ids:list[int]) -> dict[tuple[int,int],int]:
    counts={}
    for pair in zip(ids,ids[1:]):
        counts[pair]=counts.get(pair,0)+1
    return counts[pair]=pair.get(pair, 0) +1
    return counts

def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    Replaces every occurrence of `pair` with `new_id`.
    Advances by 2 when a match is found to prevent double-counting.
    """
    out = []
    i = 0
    n = len(ids)
    while i < n:
        # Check if current and next token match the target pair
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2  # Consumed both tokens in the pair
        else:
            out.append(ids[i])
            i += 1  # Not a match, keep original token
    return out


class BPETokenizer:
    def __init__(self):
        # Stores learned merge rules: (pair_a, pair_b) -> new_token_id
        # Example: {(65, 84): 256} means ('A', 'T') becomes token 256
        self.merges: dict[tuple[int, int], int] = {}

        # The Phonebook: maps token_id -> raw bytes
        # Starts with the 256 indestructible base bytes (0..255)
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Learns merge rules from training text until reaching target `vocab_size`.
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 to cover all base bytes."
        num_merges = vocab_size - 256

        # Step 1: Convert raw text into starting 8-bit bytes (0..255)
        ids = list(text.encode("utf-8"))

        # Step 2: Iteratively find top pairs and mint new token IDs
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # No more adjacent pairs to merge

            # Pick the pair that appeared most frequently
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i  # Mint ID: 256, 257, 258...

            # Replace that pair everywhere in our text sequence
            ids = merge(ids, top_pair, new_id)

            # Save the merge rule
            self.merges[top_pair] = new_id

            # Update our vocab phonebook: the new token represents the combined bytes
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose:
                print(f"Minted ID {new_id} for pair {top_pair} -> represents {self.vocab[new_id]!r}")

    def encode(self, text: str) -> list[int]:
        """
        Converts text -> token IDs.
        Always merges the lowest-ranked merge ID first to preserve hierarchy!
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)
            
            # Find which adjacent pair was learned EARLIEST during training
            # (i.e. has the smallest merge ID in self.merges)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # If none of the adjacent pairs are in our learned merges, stop!
            if pair not in self.merges:
                break

            # Apply that merge
            ids = merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Converts token IDs -> human readable string.
        """
        # 1. Look up each ID in our phonebook and glue their raw bytes together
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)

        # 2. Decode bytes back into UTF-8 text (safe against incomplete streams)
        return raw_bytes.decode("utf-8", errors="replace")


# =====================================================================
# LIVE DEMONSTRATION: Running It Right Now
# =====================================================================
if __name__ == "__main__":
    # Sample training corpus with English, Bengali, biochemistry terms, and emojis
    corpus = """
    Cellular respiration: Glucose is phosphorylated by hexokinase using ATP.
    Phosphofructokinase-1 catalyzes fructose-6-phosphate to fructose-1,6-bisphosphate.
    গ্লুকোজ এবং এটিপি বায়োকেমিক্যাল পাথওয়ে। 🧪 🔬 🧬
    """

    print("--- 1. Training the Tokenizer ---")
    tokenizer = BPETokenizer()
    # Start with 256 base bytes, and learn 20 merges -> vocab_size = 276
    tokenizer.train(corpus, vocab_size=276, verbose=True)

    print("\n--- 2. Testing Inference (Encoding New Text) ---")
    prompt = "Phosphofructokinase catalyzes glucose with ATP 🧪"
    encoded_ids = tokenizer.encode(prompt)
    print(f"Input text : {prompt}")
    print(f"Raw bytes  : {len(prompt.encode('utf-8'))} bytes")
    print(f"Token IDs  : {encoded_ids}")
    print(f"Token count: {len(encoded_ids)} tokens (Compressed!)")

    print("\n--- 3. Testing Lossless Decoding ---")
    reconstructed = tokenizer.decode(encoded_ids)
    print(f"Decoded    : {reconstructed}")
    print(f"Exact match: {reconstructed == prompt}")