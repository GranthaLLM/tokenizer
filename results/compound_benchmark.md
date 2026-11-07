Grantha compound-boundary benchmark: 258,707 compounds from 18 held-out DCS works, scored with `tools/score_compounds.py`.

| tokenizer | precision | recall (true boundaries that are token boundaries) | F1 | tokens per word |
|---|---|---|---|---|
| Grantha 128k | 0.637 | 0.730 | 0.680 | 2.73 |
| Gemma 3 / 4 | 0.350 | 0.889 | 0.503 | 5.20 |
| Sarvam-1 | 0.341 | 0.886 | 0.492 | 5.34 |
| Mistral Large 4 | 0.230 | 0.656 | 0.340 | 5.64 |
| Param2 | 0.235 | 0.580 | 0.335 | 4.99 |

Human-annotated test sets (not produced by any model):

| set | tokenizer | precision | recall | F1 | tokens per word |
|---|---|---|---|---|---|
| UoH samaasa-tagged, 7,429 words | Grantha 128k | 0.674 | 0.783 | 0.725 | 2.25 |
| | Grantha + boundary tagger | 0.782 | 0.937 | 0.853 | 2.47 |
| | Gemma 3 | 0.285 | 0.870 | 0.429 | 4.38 |
| Vedanta, 270 words | Grantha 128k | 0.422 | 0.602 | 0.496 | 4.87 |
| | Grantha + boundary tagger | 0.555 | 0.995 | 0.712 | 5.87 |
| | Gemma 3 | 0.253 | 0.778 | 0.382 | 9.34 |

UoH data: Department of Sanskrit Studies, University of Hyderabad (sanskrit.uohyd.ac.in/Corpus). Score with `--data benchmark/compounds_samasa_uoh.jsonl.gz`.

