import sys

import sentencepiece as spm

kind, vocab = sys.argv[1], int(sys.argv[2])
spm.SentencePieceTrainer.train(
    input="data/pool/sample.txt",
    model_prefix=f"data/tok_{kind}_{vocab//1000}k",
    model_type=kind,
    vocab_size=vocab,
    character_coverage=1.0,
    byte_fallback=True,
    normalization_rule_name="identity",
    split_by_unicode_script=False,
    split_digits=True,
    num_threads=8,
    input_sentence_size=2_000_000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=False,
)
