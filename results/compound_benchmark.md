Grantha compound-boundary benchmark: 258,707 compounds from 18 held-out DCS works, scored with `tools/score_compounds.py`.

| tokenizer | precision | recall (true boundaries that are token boundaries) | F1 | tokens per word |
|---|---|---|---|---|
| Grantha 128k | 0.660 | 0.769 | 0.710 | 2.75 |
| Gemma 3 / 4 | 0.350 | 0.889 | 0.503 | 5.20 |
| Sarvam-1 | 0.341 | 0.886 | 0.492 | 5.34 |
| Mistral Large 4 | 0.230 | 0.656 | 0.340 | 5.64 |
| Param2 | 0.235 | 0.580 | 0.335 | 4.99 |

Human-annotated test sets (not produced by any model):

| set | tokenizer | precision | recall | F1 | tokens per word |
|---|---|---|---|---|---|
| UoH samaasa-tagged, 7,429 words | Grantha 128k | 0.700 | 0.825 | 0.757 | 2.27 |
| | Previous release + boundary tagger | 0.782 | 0.937 | 0.853 | 2.47 |
| | Gemma 3 | 0.285 | 0.870 | 0.429 | 4.38 |
| Vedanta, 270 words | Grantha 128k | 0.591 | 0.819 | 0.686 | 4.85 |
| | Previous release + boundary tagger | 0.686 | 0.997 | 0.813 | 5.87 |
| | Gemma 3 | 0.316 | 0.894 | 0.467 | 9.34 |

UoH data: Department of Sanskrit Studies, University of Hyderabad (sanskrit.uohyd.ac.in/Corpus). Score with `--data benchmark/compounds_samasa_uoh.jsonl.gz`.


Vedanta boundaries use the same rule as the DCS set: a vowel fused into the letter before a cut is accepted on either side (`tools/vedanta_gold.py`). The tagger alone, before BPE adds its own cuts, scores P 0.85, R 0.99, F1 0.91 on this set.
