"""
Unit Tests for Byte-Level BPE Tokenizer (Chapter 5 Lab)
Verifies:
1. Exact hand-worked trace on "aaabdaaabac" (merges 256, 257, 258).
2. Non-overlapping replacement invariant (i += 2 step).
3. Round-trip lossless decode(encode(text)) == text across English, Bengali, biochemistry terms, and emojis.
4. JSON save and load serialization.
"""

import os
import tempfile
import unittest
from bpe import BPETokenizer, get_stats, merge


class TestBPETokenizer(unittest.TestCase):

    def test_01_hand_worked_trace(self):
        """
        Verify the exact 3-step hand-worked trace on 'aaabdaaabac':
        Start: [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99] (11 tokens)
        Merge 1 -> 256 = (97, 97)  -> [256, 97, 98, 100, 256, 97, 98, 97, 99] (9 tokens)
        Merge 2 -> 257 = (97, 98)  -> [256, 257, 100, 256, 257, 97, 99] (7 tokens)
        Merge 3 -> 258 = (256, 257)-> [258, 100, 258, 97, 99] (5 tokens)
        """
        text = "aaabdaaabac"
        tok = BPETokenizer()
        tok.train(text, vocab_size=256 + 3)

        self.assertIn((97, 97), tok.merges)
        self.assertEqual(tok.merges[(97, 97)], 256)
        self.assertEqual(tok.vocab[256], b"aa")

        encoded = tok.encode(text)
        expected = [258, 100, 258, 97, 99]
        self.assertEqual(encoded, expected)
        self.assertEqual(len(encoded), 5)

        decoded = tok.decode(encoded)
        self.assertEqual(decoded, text)

    def test_02_non_overlapping_replacement(self):
        """
        Verifies that [97, 97, 97] merges into [256, 97] and does NOT double-count.
        """
        ids = [97, 97, 97]
        merged = merge(ids, (97, 97), 256)
        self.assertEqual(merged, [256, 97])

    def test_03_zero_unk_and_round_trip(self):
        """
        Verifies 100% lossless round-trip across multilingual, biochemistry, and emoji text.
        Zero <unk> tokens possible because base is 0..255.
        """
        corpus = """
        Cellular respiration and glycolysis pathway:
        Glucose is phosphorylated by hexokinase or glucokinase using ATP.
        Phosphofructokinase-1 (PFK-1) catalyzes the committed step:
        Fructose-6-phosphate + ATP -> Fructose-1,6-bisphosphate + ADP.
        গ্লুকোজ এবং অ্যাডেনোসিন ট্রাইফসফেট (এটিপি) বায়োকেমিক্যাল পাথওয়ে।
        Molecular emojis: 🧪 🔬 🧬 💊.
        """
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=350)

        test_sentences = [
            "Glucose + ATP -> Glucose-6-phosphate + ADP",
            "Phosphofructokinase-1 is allosterically inhibited by high ATP levels.",
            "গ্লুকোজ বিপাক",
            "Laboratory assay: 🧪 + 🧬 = 💊",
            "Unseen synthetic molecule: Atorvastatin-d5 sodium salt",
        ]

        for s in test_sentences:
            encoded = tok.encode(s)
            # All tokens must be within the valid vocabulary
            for tid in encoded:
                self.assertIn(tid, tok.vocab)
            # Lossless round-trip
            decoded = tok.decode(encoded)
            self.assertEqual(decoded, s, f"Failed round-trip on: {s}")

    def test_04_save_and_load(self):
        """Verifies JSON serialization preserves exact merges and vocabulary."""
        text = "adenosine triphosphate ATP glycolysis phosphorylation"
        tok = BPETokenizer()
        tok.train(text, vocab_size=270)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tok.save(tmp_path)
            loaded_tok = BPETokenizer()
            loaded_tok.load(tmp_path)

            self.assertEqual(tok.merges, loaded_tok.merges)
            self.assertEqual(tok.vocab, loaded_tok.vocab)

            query = "ATP phosphorylation"
            self.assertEqual(tok.encode(query), loaded_tok.encode(query))
            self.assertEqual(tok.decode(tok.encode(query)), query)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
