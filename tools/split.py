import hashlib

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

pf = pq.ParquetFile("data/pool/pool_all.parquet")
schema = pf.schema_arrow
tw = pq.ParquetWriter("data/pool/train.parquet", schema, compression="zstd")
hw = pq.ParquetWriter("data/pool/heldout.parquet", schema, compression="zstd")
n = [0, 0]
for b in pf.iter_batches(batch_size=200000):
    t = b.to_pydict()
    keep = []
    for x in t["text"]:
        d = hashlib.blake2b(x.encode(), digest_size=2).digest()
        keep.append(int.from_bytes(d, "big") % 200 == 0)
    mask = pa.array(keep)
    tab = pa.Table.from_batches([b])
    ho, tr = tab.filter(mask), tab.filter(pc.invert(mask))
    hw.write_table(ho)
    tw.write_table(tr)
    n[0] += tr.num_rows
    n[1] += ho.num_rows
tw.close()
hw.close()
print("train", n[0], "heldout", n[1])
