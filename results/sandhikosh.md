SandhiKosh (Bhardwaj et al., LREC 2018): ashtadhyayi 1194, gita 875, internal 147, external 116, literature 146, uoh 6486 words, 8964 in total. A true boundary counts as found if a token ends at it or one letter after.

| tokenizer | precision | recall | F1 | tokens per word | F1 ashtadhyayi | F1 gita | F1 internal | F1 external | F1 literature | F1 uoh |
|---|---|---|---|---|---|---|---|---|---|---|
| Previous release + boundary tagger | 0.390 | 0.901 | 0.544 | 3.39 | 0.351 | 0.430 | 0.469 | 0.591 | 0.402 | 0.634 |
| Grantha 128k | 0.323 | 0.557 | 0.409 | 2.78 | 0.296 | 0.359 | 0.486 | 0.537 | 0.261 | 0.463 |
| Gemma 4 | 0.231 | 0.902 | 0.368 | 5.12 | 0.244 | 0.320 | 0.503 | 0.442 | 0.239 | 0.420 |
| Gemma 3 | 0.231 | 0.902 | 0.368 | 5.12 | 0.244 | 0.320 | 0.503 | 0.442 | 0.239 | 0.420 |
| Llama 4 | 0.190 | 0.948 | 0.316 | 6.56 | 0.213 | 0.279 | 0.413 | 0.356 | 0.212 | 0.357 |
| Qwen 3.5 | 0.176 | 0.961 | 0.298 | 7.22 | 0.196 | 0.251 | 0.430 | 0.360 | 0.193 | 0.341 |
| Sarvam-1 | 0.227 | 0.908 | 0.363 | 5.67 | 0.240 | 0.312 | 0.498 | 0.438 | 0.232 | 0.417 |
| Param2 | 0.226 | 0.856 | 0.357 | 5.07 | 0.248 | 0.319 | 0.456 | 0.413 | 0.243 | 0.401 |
