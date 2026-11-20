"""
End-to-End Workflow Demonstration of Byte-Level BPE Tokenizer
"""

from bpe import BPETokenizer

def run_demonstration():
    print("=" * 70)
    print("STEP 1: PREPARING THE TRAINING CORPUS")
    print("=" * 70)
    # A rich scientific training corpus
    corpus = """
    In biochemistry, glycolysis is the metabolic pathway that converts glucose into pyruvate.
    The free energy released in this process is used to form high-energy molecules ATP and NADH.
    Phosphofructokinase is a kinase enzyme that phosphorylates fructose-6-phosphate.
    The reaction produces fructose-1,6-bisphosphate.
    Hexokinase phosphorylates glucose to glucose-6-phosphate.
    Phosphorylation and dephosphorylation reactions are catalyzed by kinases and phosphatases.
    The enzyme is allosterically regulated by ATP and AMP.
    """
    print(f"Training corpus size: {len(corpus)} characters ({len(corpus.encode('utf-8'))} bytes)")

    print("\n" + "=" * 70)
    print("STEP 2: TRAINING THE BPE TOKENIZER (LEARNING MERGES)")
    print("=" * 70)
    tok = BPETokenizer()
    # Base vocab = 256 bytes. We learn 64 merges -> vocab_size = 320
    tok.train(corpus, vocab_size=320, verbose=False)
    print(f"Total learned merges: {len(tok.merges)}")
    print("Sample learned compound tokens:")
    for i, (pair, new_id) in enumerate(list(tok.merges.items())[:12]):
        token_bytes = tok.vocab[new_id]
        print(f"  Merge #{i+1:02d}: pair {pair} -> ID {new_id} ({token_bytes!r})")

    print("\n" + "=" * 70)
    print("STEP 3: TOKENIZING A LONG CLINICAL / BIOCHEMISTRY SENTENCE")
    print("=" * 70)
    sentence = "The enzyme phosphofructokinase phosphorylates fructose-6-phosphate using ATP."
    print(f"Sentence: \"{sentence}\"")

    # A: Show raw bytes
    raw_bytes = list(sentence.encode("utf-8"))
    print(f"\n[A] Raw UTF-8 bytes before BPE ({len(raw_bytes)} bytes):")
    print(f"    {raw_bytes[:15]} ... {raw_bytes[-10:]}")

    # B: Encode with BPE
    token_ids = tok.encode(sentence)
    print(f"\n[B] BPE Token IDs ({len(token_ids)} tokens):")
    print(f"    {token_ids}")

    # C: Token-by-token breakdown table
    print("\n[C] Token Breakdown Table (How the text was chunked):")
    print(f"{'Token ID':<10} | {'Chunk Bytes':<20} | {'Decoded Fragment'}")
    print("-" * 55)
    for tid in token_ids:
        chunk = tok.vocab[tid]
        decoded = chunk.decode("utf-8", errors="replace")
        print(f"{tid:<10} | {str(chunk):<20} | '{decoded}'")

    compression = (1 - len(token_ids) / len(raw_bytes)) * 100
    print("-" * 55)
    print(f"Compression: {len(raw_bytes)} raw bytes -> {len(token_ids)} tokens ({compression:.1f}% reduction!)")

    print("\n" + "=" * 70)
    print("STEP 4: LOSSLESS DECODING BACK TO HUMAN TEXT")
    print("=" * 70)
    decoded_sentence = tok.decode(token_ids)
    print(f"Decoded: \"{decoded_sentence}\"")
    print(f"Lossless match: {decoded_sentence == sentence}")
    print("=" * 70)

if __name__ == "__main__":
    run_demonstration()
