Grantha compound-boundary benchmark: 258,707 compounds from 18 held-out DCS works, scored with `tools/score_compounds.py`.

| tokenizer | precision | recall (true boundaries that are token boundaries) | F1 | tokens per word |
|---|---|---|---|---|
| Grantha 128k | 0.637 | 0.730 | 0.680 | 2.73 |
| Gemma 3 / 4 | 0.350 | 0.889 | 0.503 | 5.20 |
| Sarvam-1 | 0.341 | 0.886 | 0.492 | 5.34 |
| Mistral Large 4 | 0.230 | 0.656 | 0.340 | 5.64 |
| Param2 | 0.235 | 0.580 | 0.335 | 4.99 |
