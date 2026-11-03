Compound boundaries on held-out DCS works (238,889 compounds, a cut counts if it is at or next to the true boundary at an akshara edge) and tokens per word.

| tokenizer | cuts that are right | true boundaries respected | F1 | FLORES san | FLORES eng | DCS | GRETIL |
|---|---|---|---|---|---|---|---|
| Grantha 128k (current) | 0.585 | 0.765 | 0.663 | 1.89 | 1.17 | 1.68 | 2.05 |
| + boundary tagger (DCS gold, ByT5 teacher, akshara constraint) | 0.707 | 0.991 | 0.825 | 1.93 | 1.17 | 1.75 | 2.23 |
| Gemma 3 / 4 | 0.295 | 0.900 | 0.445 | 2.83 | 1.24 | 3.22 | 3.82 |
| Sarvam-1 | 0.285 | 0.889 | 0.431 | 2.98 | 1.43 | 3.42 | 4.21 |
| Param2 | 0.204 | 0.625 | 0.308 | 2.32 | 1.49 | 3.00 | 3.78 |
| Mistral Large 4 | 0.195 | 0.692 | 0.304 | 2.90 | 1.27 | 3.34 | 3.97 |
