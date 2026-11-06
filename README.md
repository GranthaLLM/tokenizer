# Grantha tokenizer

A joint Sanskrit (Devanagari) and English SentencePiece BPE tokenizer for GranthaLLM, 128k vocabulary. Its training text excludes everything in [`GranthaLLM/corpus`](https://github.com/GranthaLLM/corpus) and every work used as a test set below, so the corpus stays a clean evaluation set.

Status: final v0. The model is `release/grantha-128k.model` (with `grantha-128k.vocab`), and `release/hf/` holds the same tokenizer in Hugging Face format.

## Use

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="release/grantha-128k.model")
ids = sp.encode("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।")
text = sp.decode(ids)
```

or with Hugging Face `transformers`:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("release/hf")
tok.apply_chat_template([{"role": "user", "content": "अनुवादय"}], add_generation_prompt=True)
```

- **Lossless.** Byte fallback is on and whitespace is kept as written, so `decode(encode(x)) == x` for NFC text. The Hugging Face tokenizer gives identical ids to SentencePiece on all 3,036 FLORES Sanskrit, English and Hindi lines.
- **Input should be NFC.** The model was trained on NFC text. The Hugging Face tokenizer applies NFC first, the raw SentencePiece model does not.
- **Special tokens.** `<unk>` 0, `<s>` 1, `</s>` 2 (end of text), `<pad>` 3, then `<|im_start|>` 4, `<|im_end|>` 5 and 14 reserved slots `<|reserved_0|>` to `<|reserved_13|>` (6 to 19). The Hugging Face tokenizer ships a ChatML chat template.

## Results

![Tokens per Sanskrit word on four benchmarks](results/fertility.svg)

Tokens per word, lower is better. This measures cost, not quality: fewer tokens per word is cheaper to process. Grantha is trained on Sanskrit and English only, while the others are general multilingual tokenizers.

| tokenizer | FLORES Sanskrit | FLORES English | 41 corpus works | DCS held-out | GRETIL held-out |
|---|---|---|---|---|---|
| **Grantha 128k** | **1.90** | **1.17** | **1.99** | **1.68** | **2.05** |
| Param2 | 2.32 | 1.49 | 3.43 | 3.00 | 3.78 |
| Gemma 3 / 4 | 2.83 | 1.24 | 3.53 | 3.22 | 3.82 |
| Mistral Large 4 | 2.90 | 1.27 | 3.60 | 3.34 | 3.97 |
| Sarvam-1 | 2.98 | 1.43 | 3.97 | 3.42 | 4.21 |
| Llama-4 | 3.48 | 1.24 | 4.34 | 3.98 | 4.67 |
| Qwen3.5 | 4.23 | 1.26 | 5.06 | 4.47 | 5.43 |

- **FLORES-200 devtest:** 1,012 modern Sanskrit sentences.
- **41 corpus works:** the Sanskrit of `corpus/` (1.46M words), inference only.
- **DCS and GRETIL held-out works:** 18 DCS and 38 GRETIL works, chosen by hashing the work name and removed from every training source. The text is converted from IAST to Devanagari.

Sanskrit costs 1.26 times the tokens of its English translation, against 1.77 for Gemma. Full tables: [`results/fertility_bilingual.md`](results/fertility_bilingual.md) (30+ tokenizers), [`results/domain_check.md`](results/domain_check.md) (per corpus work), [`results/classical_eval.md`](results/classical_eval.md). Not measured: Kimi K3 (its tokenizer needs custom code), Tiny Aya and Apertus (gated). Gemma 4 and Sarvam-30B use the Gemma 3 tokenizer, so one row covers them.

## Example

The opening verse of Madhva's Aitareya Upanishad commentary, from the corpus. Ten words, many long compounds.

```
नारायणं निखिलपूर्णगुणैकदेहं सर्वज्ञमच्युतमपेतसमस्तदोषम् ।
प्राणस्य सर्वचिदचित्परमेश्वरस्य साक्षादधीश्वरमियां शरणं रमेशम् ॥
```

| tokenizer | tokens | per word |
|---|---|---|
| **Grantha 128k** | **29** | 2.9 |
| Gemma 3 / 4 | 47 | 4.7 |
| Param2 | 50 | 5.0 |
| Sarvam-1 | 51 | 5.1 |
| Mistral Large 4 | 53 | 5.3 |
| Llama-4 | 62 | 6.2 |
| Qwen3.5 | 72 | 7.2 |

Grantha: `नारायणं | निखिल | पूर्ण | गुण | ैक | देहं | सर्वज्ञ | म | च्यु | तम | पेत | समस्त | दोषम् | । | प्राणस्य | सर्व | चिद | चि | त्परमे | श्वरस्य | साक्षाद | धी | श्वर | मि | यां | शरणं | रमे | शम् | ॥`

Gemma 3 / 4: `न | ारा | यण | ं | निखिल | पूर्ण | गुण | ैक | देह | ं | सर्व | ज्ञ | म | च | ्यु | तम | प | ेत | सम | स्त | द | ोष | म् | । | ...`

Grantha keeps whole words such as `नारायणं`, `निखिल`, `पूर्ण`, `सर्वज्ञ` and `प्राणस्य` intact, where the others break them into letters and bare vowel signs. It is not perfect: long compounds are cut at the wrong place in several spots, for example `सर्वज्ञमच्युतमपेत…`.

## Compound splitting

Sandhi and compounds hide where words end, and BPE cuts at frequent letter strings instead. On 238,889 compounds from 18 held-out DCS works, 76.5% of true boundaries are also token boundaries, and boundary F1 is 0.66. Gemma 3 / 4 respects 90.0% but needs about 1.5 times the tokens, because it cuts everything into small pieces. Param2 respects 62.5%. Long Vedantic compounds are the weak spot, for example `सृष्टिस्थित्यप्ययेहानियति…` comes out with `हानि | यति` glued together.

An experimental boundary tagger raises this to 97.2% respected and F1 0.86, at 3 to 12% more tokens. Marks are only needed on about 11% of words. A small language model trained with it ties with the plain tokenizer within 0.6% bits per byte (one seed, 91M parameters), so correct splitting costs no model quality at that scale, and no gain was shown. It is not part of the release. The code is in `tools/` and the numbers are in `results/tagged_eval.md` and `results/proxy_*.json`.

## Variants

| training data | vocab | FLORES Sanskrit | FLORES English | DCS | GRETIL |
|---|---|---|---|---|---|
| web pool + DCS + GRETIL (release) | 128k | 1.90 | 1.17 | 1.68 | 2.05 |
| same, digits not split | 128k | 1.86 | 1.13 | 1.68 | 2.06 |
| web pool + DCS | 128k | 1.87 | 1.16 | 1.71 | 2.15 |
| web pool only | 128k | 1.87 | 1.16 | 1.75 | 2.17 |
| web pool + DCS + GRETIL | 64k | 2.06 | 1.24 | 1.89 | 2.27 |
| web pool only | 64k | 2.05 | 1.23 | 1.97 | 2.38 |

The Sanskrit share of the training mix barely matters between 30% and 70% (64k runs), so the final mix is 50/50. Making the danda a standalone piece costs 2 to 5% more Sanskrit tokens. Leaving digits unsplit saves 2% of Sanskrit and 3% of English tokens on FLORES, but multi-digit tokens are known to hurt numeric reasoning, so the release splits digits.

## Training data

| part | source | notes |
|---|---|---|
| Sanskrit, web | [Sansar](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus), permissive configs (CC0, CC-BY-4.0, ODC-By, MIT, Apache-2.0) | 2.31M lines, 186.6M words after cleaning. Mostly Sangraha and IndicCorp web and scan text |
| Sanskrit, classical | [DCS](https://github.com/OliverHellwig/sanskrit) (CC BY 4.0), [GRETIL](https://github.com/INDOLOGY/GRETIL-mirror) Sanskrit TEI files (CC BY-NC-SA 4.0) | 253 DCS works (3.0M words) and 738 GRETIL works (12.5M words) after holdout, `corpus/` overlap and duplicate removal. IAST converted to Devanagari with `tools/iast.py` |
| English | Wikipedia (20231101.en, shard 0), FineWeb-Edu (sample/10BT shard 0) | 400M characters in the mix |

Cleaning: Unicode NFC, invisible characters stripped, a Devanagari-share filter, removal of any line sharing a 20-character window with `corpus/` Sanskrit (about 4% of lines) or with a held-out work, exact line deduplication. FLORES overlap with training text was 1 Sanskrit and 123 English lines, removed.

Licence note: Wikipedia is CC BY-SA 4.0 and every GRETIL Sanskrit file is CC BY-NC-SA 4.0. This does not matter for internal benchmarking. It does for any public release.

## Limits

- Fertility is a cost measure. No full-scale model has been trained with this tokenizer, so there is no quality result beyond the small test above.
- No Sanskrit training line shares a run of 27 or more characters with `corpus/` or with a held-out work. Shorter shared phrases, such as common formulae, can still occur.
- DCS and GRETIL text is transliterated by our own converter and keeps editorial conventions such as `प्य् उक्तं`, so some of it is split at sandhi in a way fused text is not.
- Tokens per word is a weak unit for Sanskrit because sandhi fuses words, and it is noisy on GRETIL because some files are set without spaces.
- Long compounds are cut at the wrong place in places (see Compound splitting).
- Danda isolation and digit splitting were ablated (see Variants). NFC-only normalisation and the 50/50 Sanskrit and English mix were chosen, with the mix tested only at 64k.

## Credits

- [Digital Corpus of Sanskrit](https://github.com/OliverHellwig/sanskrit) (Oliver Hellwig), CC BY 4.0, training text and the held-out test works.
- [Sansar](https://huggingface.co/datasets/MuseMesh/sansar-sanskrit-corpus), which collects [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (AI4Bharat), IndicCorp and others. [GRETIL](https://github.com/INDOLOGY/GRETIL-mirror).
- [FLORES-200](https://github.com/facebookresearch/flores) (Meta) and [SandhiKosh](https://github.com/sanskrit-sandhi/SandhiKosh) (Bhardwaj et al., LREC 2018) for evaluation.
- The experimental tagger uses labels from [ByT5-Sanskrit](https://arxiv.org/abs/2409.13920) (Nehrdich, Hellwig and Keutzer, EMNLP Findings 2024); the paper states Apache 2.0.
