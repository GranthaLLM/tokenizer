import sys
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import SentencePieceExtractor

model, out = sys.argv[1], sys.argv[2]
proto = pb.ModelProto()
proto.ParseFromString(open(model, "rb").read())
ex = SentencePieceExtractor(model).extract(BPE)
vocab, merges = ex["vocab"], ex["merges"]
unk = proto.pieces[proto.trainer_spec.unk_id].piece
special = [p.piece for p in proto.pieces if p.type in (2, 3, 4)]
user = [p.piece for p in proto.pieces if p.type == 4]
tok = Tokenizer(BPE(vocab, merges, unk_token=unk, fuse_unk=True, byte_fallback=True, dropout=None))
tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
tok.pre_tokenizer = None
tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(content=" ", left=1, right=0)])
bos, eos = proto.pieces[proto.trainer_spec.bos_id].piece, proto.pieces[proto.trainer_spec.eos_id].piece
pad = proto.pieces[proto.trainer_spec.pad_id].piece if proto.trainer_spec.pad_id >= 0 else None
tok.add_special_tokens([t for t in [unk, bos, eos, pad] if t])
if user:
    tok.add_special_tokens(user)
kw = dict(tokenizer_object=tok, unk_token=unk, bos_token=bos, eos_token=eos)
if pad:
    kw["pad_token"] = pad
hf = PreTrainedTokenizerFast(**kw)
Path(out).mkdir(parents=True, exist_ok=True)
hf.save_pretrained(out)
sp = spm.SentencePieceProcessor(model_file=model)
F = "data/flores/flores200_dataset/devtest/"
bad = tot = rt = 0
first = None
for l in ("san_Deva", "eng_Latn", "hin_Deva"):
    for s in Path(F + l + ".devtest").read_text(encoding="utf-8").splitlines():
        tot += 1
        a, b = sp.encode(s), hf.encode(s, add_special_tokens=False)
        if a != b:
            bad += 1
            first = first or (s[:50], a[:10], b[:10])
        if hf.decode(b) == sp.decode(a):
            rt += 1
print("vocab", len(hf), "| ids equal to SentencePiece:", tot - bad, "of", tot, "| decode equal:", rt, "of", tot)
if first:
    print("first mismatch:", first)
