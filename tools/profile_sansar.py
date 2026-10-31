import glob
import re
from collections import defaultdict

import pyarrow.parquet as pq

dev = re.compile(r"[ऀ-ॿ]")
agg = defaultdict(lambda: dict(rows=0, words=0, chars=0, dev=0, lic=set()))
for f in sorted(glob.glob("data/raw/sansar/data/*/*.parquet")):
    pf = pq.ParquetFile(f)
    for b in pf.iter_batches(batch_size=50000, columns=["text", "source", "licence"]):
        for t, s, l in zip(*(b.column(i).to_pylist() for i in range(3))):
            a = agg[s]
            a["rows"] += 1
            a["words"] += len(t.split())
            a["chars"] += len(t)
            a["dev"] += len(dev.findall(t))
            a["lic"].add(l)
print(f"{'source':18}{'rows':>9}{'Mwords':>9}{'chars/row':>10}{'dev%':>6}  licences")
for s, a in sorted(agg.items(), key=lambda x: -x[1]["words"]):
    print(f"{s:18}{a['rows']:9}{a['words']/1e6:9.2f}{a['chars']/a['rows']:10.0f}{100*a['dev']/max(a['chars'],1):6.0f}  {','.join(sorted(a['lic']))}")
