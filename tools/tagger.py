import json
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from akshara import bounds

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXLEN = 96


class Net(nn.Module):
    def __init__(self, vocab, emb=128, hid=192):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.rnn = nn.LSTM(emb, hid, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(2 * hid, 1)

    def forward(self, x):
        return self.out(self.rnn(self.emb(x))[0]).squeeze(-1)


def encode(words, vocab):
    n = min(max(len(w) for w in words), MAXLEN)
    x = torch.zeros(len(words), n, dtype=torch.long)
    for i, w in enumerate(words):
        for j, c in enumerate(w[:n]):
            x[i, j] = vocab.get(c, 1)
    return x


def mask_for(words, n):
    m = torch.zeros(len(words), n, dtype=torch.bool)
    for i, w in enumerate(words):
        for b in bounds(w[:MAXLEN]):
            if 0 < b < len(w) and b - 1 < n:
                m[i, b - 1] = True
    return m


class Tagger:
    def __init__(self, path="data/tagger.pt", vocab_path="data/tagger_vocab.json", threshold=0.5, hid=192):
        self.vocab = json.load(open(vocab_path, encoding="utf-8"))
        self.net = Net(len(self.vocab) + 2, hid=hid).to(DEV)
        self.net.load_state_dict(torch.load(path, map_location=DEV))
        self.net.eval()
        self.th = threshold
        self.cache = {}
        self.override = {}

    @torch.no_grad()
    def cuts_many(self, words, batch=2048):
        todo = [w for w in dict.fromkeys(words) if w not in self.cache]
        for w in [w for w in todo if w in self.override]:
            self.cache[w] = self.override[w]
        todo = [w for w in todo if w not in self.cache]
        todo.sort(key=len)
        for i in range(0, len(todo), batch):
            ws = todo[i : i + batch]
            x = encode(ws, self.vocab).to(DEV)
            p = torch.sigmoid(self.net(x)).cpu()
            m = mask_for(ws, x.shape[1])
            for k, w in enumerate(ws):
                self.cache[w] = [j + 1 for j in range(min(len(w), x.shape[1])) if m[k, j] and p[k, j] > self.th and j + 1 < len(w)]
        return [self.cache[w] for w in words]

    def cuts(self, w):
        return set(self.cuts_many([w])[0])


def synth(tr, n, rng):
    priors, finals = [], []
    for w, cuts in tr:
        if cuts:
            segs = [w[a:b] for a, b in zip([0] + list(cuts), list(cuts) + [len(w)])]
            priors += [x for x in segs[:-1] if len(x) >= 2 and x[-1] not in "्ंः"]
            finals.append(segs[-1])
        elif len(w) >= 3:
            finals.append(w)
    priors, finals = list(set(priors)), list(set(finals))
    out = set()
    while len(out) < n:
        k = rng.randint(2, 6)
        parts = [rng.choice(priors) for _ in range(k - 1)] + [rng.choice(finals)]
        w = "".join(parts)
        if len(w) > 90:
            continue
        cuts, pos = [], 0
        for x in parts[:-1]:
            pos += len(x)
            cuts.append(pos)
        out.add((w, tuple(cuts)))
    return list(out)


def train(out="data/tagger2.pt", vocab_out="data/tagger2_vocab.json", hid=256, n_synth=900000, epochs=8, extra=None, long_boost=1):
    random.seed(0)
    torch.manual_seed(0)
    data = pickle.load(open("data/dcs_tagger_train.pkl", "rb"))
    items = list(data.keys())
    random.shuffle(items)
    dev_items, tr = items[:20000], items[20000:]
    if extra:
        tr = list(dict.fromkeys(tr + list(pickle.load(open(extra, "rb")).keys())))
    if n_synth:
        tr = tr + synth(tr, n_synth, random.Random(1))
    if long_boost > 1:
        tr = tr + [it for it in tr if len(it[0]) >= 24 and it[1]] * (long_boost - 1)
    chars = sorted({c for w, _ in tr for c in w})
    vocab = {c: i + 2 for i, c in enumerate(chars)}
    json.dump(vocab, open(vocab_out, "w", encoding="utf-8"), ensure_ascii=False)
    net = Net(len(vocab) + 2, hid=hid).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    posw = torch.tensor(2.0, device=DEV)
    EPOCHS, BS = epochs, 512
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * (len(tr) // BS + 64))
    for ep in range(EPOCHS):
        net.train()
        random.shuffle(tr)
        chunks = [tr[i : i + 8192] for i in range(0, len(tr), 8192)]
        t0, tot, nb = time.time(), 0.0, 0
        for ch in chunks:
            ch.sort(key=lambda it: len(it[0]))
            for i in range(0, len(ch), BS):
                b = ch[i : i + BS]
                ws = [w for w, _ in b]
                x = encode(ws, vocab).to(DEV)
                y = torch.zeros(x.shape, device=DEV)
                for k, (w, t) in enumerate(b):
                    for c in t:
                        if c - 1 < x.shape[1]:
                            y[k, c - 1] = 1
                valid = (x != 0).float()
                loss = (nn.functional.binary_cross_entropy_with_logits(net(x), y, pos_weight=posw, reduction="none") * valid).sum() / valid.sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
                nb += 1
        torch.save(net.state_dict(), out)
        tg = Tagger(out, vocab_out, hid=hid)
        tp = fp = fn = 0
        for w, t in dev_items[:8000]:
            c = set(tg.cuts(w))
            tp += len(c & set(t))
            fp += len(c - set(t))
            fn += len(set(t) - c)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"epoch {ep+1} loss {tot/nb:.4f} dev P {p:.3f} R {r:.3f} F1 {2*p*r/max(p+r,1e-9):.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "t6":
        train("data/tagger6.pt", "data/tagger6_vocab.json", hid=256, n_synth=0, epochs=10, extra="data/teacher_train.pkl", long_boost=3)
    elif len(sys.argv) > 1 and sys.argv[1] == "t5":
        train("data/tagger5.pt", "data/tagger5_vocab.json", hid=192, n_synth=0, epochs=8, extra="data/teacher_train.pkl")
    elif len(sys.argv) > 1 and sys.argv[1] == "teacher":
        train("data/tagger4.pt", "data/tagger4_vocab.json", hid=192, n_synth=0, epochs=8, extra="data/teacher_train.pkl")
    elif len(sys.argv) > 1 and sys.argv[1] == "uoh":
        train("data/tagger3.pt", "data/tagger3_vocab.json", hid=192, n_synth=0, epochs=8, extra="data/uoh_train.pkl")
    else:
        train()
