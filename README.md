# Grantha tokenizer

A joint Sanskrit (Devanagari) and English SentencePiece BPE tokenizer for GranthaLLM. It is trained on a text pool that excludes everything in [`GranthaLLM/corpus`](https://github.com/GranthaLLM/corpus), so the corpus stays a clean evaluation set.

Status: v0 candidate. The numbers below are measured, the tokenizer is not yet frozen or released. See [Limits](#limits).

## Results

Tokens per word on FLORES-200 devtest (1,012 sentences), lower is better.

| tokenizer | Sanskrit | English | Sanskrit / English tokens |
|---|---|---|---|
| **Grantha joint 50/50, 128k** | **1.87** | **1.16** | 1.25 |
| Grantha joint 50/50, 64k | 2.05 | 1.23 | 1.29 |
| Grantha Sanskrit-only, 128k | 1.76 | 5.82 | n/a |
| Gemma-3 | 2.83 | 1.24 | 1.77 |
| Sarvam-1 | 2.98 | 1.43 | 1.61 |
| Sarvam-M | 3.75 | 1.27 | 2.29 |
| Llama-3 | 4.22 | 1.24 | 2.64 |
| Qwen3 | 7.34 | 1.26 | 4.51 |

Full table with tokens per character and held-out columns: [`results/fertility_bilingual.md`](results/fertility_bilingual.md).

Sandhi boundaries on [SandhiKosh](https://github.com/sanskrit-sandhi/SandhiKosh) (8,964 single-word gold splits). A cut counts if it falls at the gold point or one character after. Because a tokenizer that cuts more often scores higher by luck, the table also gives a random baseline with the same tokens per word.

| tokenizer | F1 | random F1 at same cut count | F1 above random |
|---|---|---|---|
| Grantha joint 128k | 0.416 | 0.235 | +0.18 |
| Grantha joint 64k | 0.417 | 0.245 | +0.17 |
| Gemma-3 | 0.368 | 0.279 | +0.09 |
| Sarvam-1 | 0.363 | 0.271 | +0.09 |

The gold labels are the underlying split, so the cut position is approximated by common prefix. Treat the numbers as relative.

Other findings:

- The Sanskrit share of the training mix barely matters between 30% and 70% (64k runs: Sanskrit 2.15, 2.05, 1.99; English 1.20, 1.23, 1.27).
- Making the danda a standalone piece costs 2 to 5% more Sanskrit tokens.
- The learned 128k vocabulary has no mixed-script or OCR-garbage pieces. A few pieces come from repeated headings, for example `द्वितीयोऽध्यायः`.
- FLORES overlap with the training text was 1 Sanskrit and 123 English lines. They were removed.

## Training data

| part | source | notes |
|---|---|---|
| Sanskrit | [Sansar](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus), permissive configs (CC0, CC-BY-4.0, ODC-By, MIT, Apache-2.0) | 2.31M lines, 186.6M words after cleaning. Mostly Sangraha and IndicCorp web and scan text |
| English | Wikipedia (20231101.en, shard 0), FineWeb-Edu (sample/10BT shard 0) | 600M characters each before mixing |

Pool steps, in `tools/`: Unicode NFC and invisible-character stripping, a Devanagari-share filter, removal of any line that shares a 20-character window with a Sanskrit string in `corpus/` (about 4% of lines), then exact line deduplication. One line in 200 is held out.

Licence note: Wikipedia is CC BY-SA 4.0. The Sanskrit side is permissive only, the English side is not. Decide this before releasing the tokenizer.

## Limits

- Sanskrit is measured on FLORES, which is modern news prose. No measurement yet on shastra or commentary text.
- English is measured on FLORES only, not on Vedanta-style English.
- The Sanskrit pool lacks the cleaner scholarly sources (DCS, GRETIL). Classical vocabulary is likely under-represented.
- No `tokenizer.json`, special tokens or round-trip tests yet.
- Digit splitting, NFC-only normalisation and danda handling were chosen, not ablated.
- Parity-aware BPE and vocabulary extension of an existing tokenizer were not run.
- Tokens per word is a weak unit for Sanskrit because sandhi fuses words. Per-character numbers are in the results table.

## Layout

```
tools/
  fetch_sansar.py, fetch_english.py   download data
  profile_sansar.py                   per-source stats
  common.py                           normalisation and line filters
  holdout_index.py                    fingerprint corpus/ Sanskrit
  build_pool.py                       clean, guard against corpus/, dedupe
  split.py, make_sample.py            held-out slice, balanced training sample
  train_spm.py                        Sanskrit-only BPE
  make_mix.py, train_mix.py           joint Sanskrit and English BPE
  measure.py                          tokens per word, baselines
  sandhi_eval.py                      SandhiKosh boundary test
results/                              committed tables
```
