# Tasks for the 3090 machine

You are working on the Grantha Sanskrit tokenizer repo, branch `pc`. This machine has an NVIDIA RTX 3090 (24 GB). Two GPU jobs are queued. Do them in the order below, and report results back.

Rules: commits are authored as the repo owner (use the repo's git identity, do not set an author), no co-author lines, conventional commit subjects of at most 50 characters, no em dashes anywhere in code, comments or messages. Do not change the training or evaluation logic. Only fix bugs that stop the scripts from running, and say what you changed.

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pc.txt     # use the CUDA build of torch for this GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If CUDA is not available, stop and tell the user.

## 1. Proxy-model test (decisive result, run first)

Goal: see if better compound boundaries help a language model. Three tokenizers are compared by training the same small GPT (about 90M parameters) on the same text, then measuring bits per byte on held-out text. Bits per byte is comparable across tokenizers, token-level perplexity is not.

The token files are NOT in git. The user copies them from the Mac into `data/proxy/` (about 570 MB). Check these exist before starting:

```
data/proxy/meta.json
data/proxy/{current,akshara,tagged}_train.npy
data/proxy/{current,akshara,tagged}_val_{dcs,gretil,corpus,english}.npy
```

If they are missing, stop and ask the user to run, on the Mac: `rsync -av data/proxy/ <pc>:<repo path>/data/proxy/`

Run, one after another (each takes roughly 30 to 60 minutes, watch the tok/s line):

```bash
python tools/proxy_lm.py current
python tools/proxy_lm.py akshara
python tools/proxy_lm.py tagged
```

If you hit out-of-memory, add `--bs 32` to all three runs (keep it identical across them). Do not change any other setting between runs. Each run writes `results/proxy_<name>.json`.

What the names mean: `current` is the current Grantha 128k tokenizer, `akshara` keeps vowel signs attached to their letters, `tagged` adds the compound boundary tagger.

When all three finish, print a table with one row per tokenizer and one column per validation set (dcs, gretil, corpus, english) of bits per byte, plus training tokens and steps. Lower is better. Then commit `results/proxy_*.json` and push the branch:

```bash
git add results/proxy_*.json && git commit -m "results: proxy model bits per byte" && git push
```

If the push fails because of credentials, just show the table and tell the user.

## 2. Sandhi-resolution labeling (long job, run after step 1, can run overnight)

This labels about 2.8M Sanskrit words (the most frequent corpus words, 73% of corpus word occurrences) with the ByT5-Sanskrit segmenter. The labels are the sandhi-resolved words that the new tokenizer trains on. It downloads a 2.3 GB model from Hugging Face on first run. The script is resumable.

```bash
gunzip -k data/resolve_words.txt.gz
tmux new -s teacher      # or screen, or nohup
python tools/teacher_gpu.py data/resolve_words.txt data/resolve_labels.pkl --chunk 20000 --max-bytes 200000
```

After the first chunk prints a words per second line, report that number to the user. Expect hundreds of words per second or better on this GPU. If you hit out-of-memory, lower `--max-bytes` (default 40000). If it is under 100 words per second, raise it.

When it finishes, tell the user the output is `data/resolve_labels.pkl` (do not commit it, it is large) and that it should be copied back to the Mac:

```bash
rsync -av <pc>:<repo path>/data/resolve_labels.pkl data/
```

## What to report

1. GPU name and whether the three proxy runs finished.
2. The bits per byte table.
3. The labeling speed and whether it finished.
4. Anything you had to change to make the scripts run.
