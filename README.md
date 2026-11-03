# Grantha tokenizer

A joint Sanskrit (Devanagari) and English SentencePiece BPE tokenizer for GranthaLLM. Its training text excludes everything in [`GranthaLLM/corpus`](https://github.com/GranthaLLM/corpus) and every work used as a test set below, so the corpus stays a clean evaluation set.

Status: v0 candidate, 128k vocabulary. Measured, not yet frozen or released. See [Limits](#limits).

## Results

![Tokens per Sanskrit word on four benchmarks](results/fertility.svg)

This measures cost, not quality: fewer tokens per word is cheaper to process, and no model has been trained on the tokenizer yet. Grantha is trained on Sanskrit and English only, while the others are general multilingual tokenizers. Grantha needs about 33% fewer tokens than the best general-purpose tokenizers (Gemma 3 / 4, Mistral Large 4) on FLORES-200 and 44 to 48% fewer on the corpus and classical works. Against Param2, the closest India-focused rival, the gap is 19% on FLORES and 42 to 46% on the other three. The benchmarks:

- **FLORES-200 devtest:** 1,012 modern Sanskrit sentences.
- **41 corpus works:** the Sanskrit of `corpus/` (1.46M words), inference only.
- **DCS and GRETIL held-out works:** 18 DCS and 38 GRETIL works, chosen by hashing the work name and removed from every training source. The text is converted from IAST to Devanagari.

English is not sacrificed: 1.17 tokens per word on FLORES, against 1.24 for Gemma 3 / 4 and Llama-4, 1.26 for Qwen3.5, 1.27 for Mistral Large 4 and 1.49 for Param2. Sanskrit costs 1.25 times the tokens of its English translation, against 1.77 for Gemma.

Full tables: [`results/fertility_bilingual.md`](results/fertility_bilingual.md) (FLORES, 30+ tokenizers), [`results/domain_check.md`](results/domain_check.md) (per corpus work), [`results/classical_eval.md`](results/classical_eval.md).

Tokenizers not measured: Kimi K3 (its tokenizer needs custom code), Tiny Aya and Apertus (gated). Gemma 4 and Sarvam-30B use the Gemma 3 tokenizer, so one row covers them.

## Example

The opening verse of Madhva's Aitareya Upanishad commentary, from the corpus. Ten words, many long compounds. Each `|` is a cut between tokens.

```
नारायणं निखिलपूर्णगुणैकदेहं सर्वज्ञमच्युतमपेतसमस्तदोषम् ।
प्राणस्य सर्वचिदचित्परमेश्वरस्य साक्षादधीश्वरमियां शरणं रमेशम् ॥
```

| tokenizer | tokens | per word |
|---|---|---|
| **Grantha 128k** | **29** | 2.9 |
| Param2 | 50 | 5.0 |
| Gemma 3 / 4 | 47 | 4.7 |
| Mistral Large 4 | 53 | 5.3 |
| Sarvam-1 | 51 | 5.1 |
| Llama-4 | 62 | 6.2 |
| Qwen3.5 | 72 | 7.2 |

- **Grantha 128k, 29 tokens:** नारायणं | निखिल | पूर्ण | गुण | ैक | देहं | सर्वज्ञ | म | च्यु | तम | पेत | समस्त | दोषम् | । | प्राणस्य | सर्व | चिद | चि | त्परमे | श्वरस्य | साक्षाद | धी | श्वर | मि | यां | शरणं | रमे | शम् | ॥

- **Param2, 50 tokens:** नार | ाय | णं | निख | िल | पूर्ण | गुण | ैक | देह | ं | सर्व | ज्ञ | म | च | ्य | ुत | म | प | ेत | सम | स्त | दोष | म् | · | । | प्राण | स्य | सर्व | च | िद | चित | ्प | रम | ेश | ्वर | स्य | साक्ष | ाद | धी | श्वर | म | ियां | श | रणं | र | म | ेश | म् | � | �

- **Gemma 3 / 4, 47 tokens:** न | ारा | यण | ं | निखिल | पूर्ण | गुण | ैक | देह | ं | सर्व | ज्ञ | म | च | ्यु | तम | प | ेत | सम | स्त | द | ोष | म् | । | प्राण | स्य | सर्व | चि | द | चित | ् | पर | म | ेश्वर | स्य | सा | क्षा | द | धी | श्वर | म | ियां | शरण | ं | रमेश | म् | ॥

<details><summary>The other four tokenizers</summary>

- **Mistral Large 4, 53 tokens:** न | ार | ायण | ं | न | िख | िल | पूर्ण | गु | ण | ैक | द | ेह | ं | सर्व | ज्ञ | म | च | ्य | ुत | म | प | ेत | सम | स्त | द | ोष | म् | । | प्राण | स्य | सर्व | च | िद | चित | ्प | र | म | ेश्वर | स्य | साक्ष | ाद | धी | श्वर | म | ियां | श | रण | ं | रम | ेश | म् | ॥

- **Sarvam-1, 51 tokens:** नारायण | ं | नि | ख | िल | पूर्ण | गु | ण | ैक | देह | ं | सर्व | ज्ञ | म | च्यु | तम | प | ेत | सम | स्त | दो | ष | म् | । | प्राण | स्य | सर्व | चि | द | चित | ् | पर | म | ेश्वर | स्य | साक्षा | द | धी | श्वर | म | ियां | श | रण | ं | रम | ेश | म् | · | � | � | �

- **Llama-4, 62 tokens:** न | ार | ाय | ण | ं | न | िख | िल | पूर्ण | ग | ुण | ैक | द | ेह | ं | सर्व | ज्ञ | म | च | ्य | ुत | म | प | ेत | सम | स्त | द | ोष | म | ् | । | प्र | ाण | स्य | सर्व | च | िद | चित | ्प | र | म | ेश्वर | स्य | सा� | � | ्ष | ाद | धी | श | ्वर | म | ियां | श | रण | ं | र | म | ेश | म | ् | � | �

- **Qwen3.5, 72 tokens:** न | ारा | य | ण | ं | नि | ख | िल | पूर्ण | ग | ु | ण | ै | क | दे | ह | ं | सर्व | ज्ञ | म | च | ्य | ु | त | म | प | ेत | स | म | स्त | द | ो | ष | म् | । | प्रा | ण | स | ्य | सर्व | च | ि | द | चित | ् | पर | म | ेश | ्व | र | स | ्य | सा | क्षा | द | धी | श | ्व | र | म | िया | ं | श | रण | ं | र | म | ेश | म् | · | � | �

</details>

Grantha keeps whole words such as `नारायणं`, `निखिल`, `पूर्ण`, `सर्वज्ञ` and `प्राणस्य` intact, where the others break them into letters and bare vowel signs. It is not perfect: long compounds are cut at the wrong place in several spots, for example `सर्वज्ञमच्युतमपेत…`, which runs on past the word boundaries.

## Variants

| training data | vocab | FLORES Sanskrit | FLORES English | DCS | GRETIL |
|---|---|---|---|---|---|
| web pool + DCS + GRETIL | 128k | 1.89 | 1.17 | 1.68 | 2.05 |
| web pool + DCS | 128k | 1.87 | 1.16 | 1.71 | 2.15 |
| web pool only | 128k | 1.87 | 1.16 | 1.75 | 2.17 |
| web pool + DCS + GRETIL | 64k | 2.06 | 1.24 | 1.89 | 2.27 |
| web pool only | 64k | 2.05 | 1.23 | 1.97 | 2.38 |

The Sanskrit share of the training mix barely matters between 30% and 70% (64k runs), so the final mix is 50/50. Making the danda a standalone piece costs 2 to 5% more Sanskrit tokens.

Adding DCS and GRETIL lowers classical fertility by 4 to 6% and costs almost nothing elsewhere. Your corpus works go from 1.99 to 1.98.

## Sandhi boundaries

[SandhiKosh](https://github.com/sanskrit-sandhi/SandhiKosh), 8,964 single-word gold splits. A cut counts if it falls at the gold point or one character after. A tokenizer that cuts more often scores higher by luck, so the table includes a random baseline with the same cut count.

| tokenizer | F1 | random F1 at same cut count | F1 above random |
|---|---|---|---|
| Grantha 128k, web pool only | 0.416 | 0.235 | +0.18 |
| Grantha 128k, + DCS + GRETIL | 0.407 | 0.235 | +0.17 |
| Gemma 3 / 4 | 0.368 | 0.279 | +0.09 |
| Sarvam-1 | 0.363 | 0.271 | +0.09 |

The gold labels are the underlying split, so the cut position is approximated by common prefix. Treat the numbers as relative.

## Compound boundaries

Sanskrit compounds and sandhi hide where words end, and plain BPE cuts at frequent letter strings instead. We measure this on 238,889 compounds from 18 DCS works held out of every training source. A cut counts as right if it is at, or next to, the true boundary at an akshara edge, because sandhi can fuse a vowel into the letter before it.

| tokenizer | cuts that are right | true boundaries respected | F1 | tokens per word, FLORES / DCS / GRETIL |
|---|---|---|---|---|
| Grantha 128k | 0.585 | 76.5% | 0.663 | 1.89 / 1.68 / 2.05 |
| Grantha 128k + boundary tagger | 0.707 | 99.1% | 0.825 | 1.93 / 1.75 / 2.23 |
| Gemma 3 / 4 | 0.295 | 90.0% | 0.445 | 2.83 / 3.22 / 3.82 |
| Param2 | 0.204 | 62.5% | 0.308 | 2.32 / 3.00 / 3.78 |

The boundary tagger variant puts an invisible mark at compound and sandhi joints before tokenization, and BPE is trained so no piece crosses a mark. The marks are dropped from the output, so decoding returns the original text. The tagger is a small character model trained on DCS gold splits and on labels from [ByT5-Sanskrit](https://aclanthology.org/2024.findings-emnlp.805.pdf) for 263k Vedanta-style, Bhagavata and pool words. The tokenizer then also needs the tagger at use time.

Known gaps:

- A piece can still hold a fused akshara together with its neighbour: `ईहा नियति` comes out as `हानि | यति`, because the vowel fuses into `हा`.
- The tagger is a student of ByT5-Sanskrit and misses some very long compounds. ByT5-Sanskrit itself splits them correctly but runs at about 35 to 90 words per second on a laptop.
- DCS is mostly epic and Purana text, and the labels outside it come from ByT5, not from people. There is no human-checked Vedanta compound test yet.
- UoH sandhi splits mark only sandhi joints, not whole compounds, so they were not used as gold for compounds.

## Training data

| part | source | notes |
|---|---|---|
| Sanskrit, web | [Sansar](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus), permissive configs (CC0, CC-BY-4.0, ODC-By, MIT, Apache-2.0) | 2.31M lines, 186.6M words after cleaning. Mostly Sangraha and IndicCorp web and scan text |
| Sanskrit, classical | [DCS](https://github.com/OliverHellwig/sanskrit) (CC BY 4.0), [GRETIL](https://github.com/INDOLOGY/GRETIL-mirror) Sanskrit TEI files (CC BY-NC-SA 4.0) | 253 DCS works (3.0M words) and 738 GRETIL works (12.5M words) after holdout, `corpus/` overlap and duplicate removal. IAST converted to Devanagari with `tools/iast.py` |
| English | Wikipedia (20231101.en, shard 0), FineWeb-Edu (sample/10BT shard 0) | 400M characters in the mix |

Cleaning: Unicode NFC, invisible characters stripped, a Devanagari-share filter, removal of any line sharing a 20-character window with `corpus/` Sanskrit (about 4% of lines) or with a held-out work, exact line deduplication. FLORES overlap with training text was 1 Sanskrit and 123 English lines, removed.

Licence note: Wikipedia is CC BY-SA 4.0 and every GRETIL Sanskrit file is CC BY-NC-SA 4.0. This does not matter for internal benchmarking. It does for any public release of the +GRETIL variant.

## Limits

- Fertility is a cost measure. No model has been trained with this tokenizer yet, so there is no quality result.
- About 2% of training lines share a short phrase (a 20-character window) with `corpus/`. These are formulae, not passages.
- DCS and GRETIL text is transliterated by our own converter and keeps editorial conventions such as `प्य् उक्तं`, so some of it is split at sandhi in a way fused text is not.
- Tokens per word is a weak unit for Sanskrit because sandhi fuses words, and it is noisy on GRETIL because some files are set without spaces.
- Long compounds are the weak spot (see Compound boundaries). Ordinary words are fine: 71% stay whole.
- No `tokenizer.json`, special tokens or round-trip tests yet.
- Digit splitting, NFC-only normalisation and danda handling were chosen, not ablated.
- Parity-aware BPE and vocabulary extension of an existing tokenizer were not run.
