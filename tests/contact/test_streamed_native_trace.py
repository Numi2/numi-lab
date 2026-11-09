#!/usr/bin/env python3
"""Exercise large, truncated, malformed and chunk-boundary native streams."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import tracemalloc
from streamed_native_trace import _Reader, read_native

for chunk in (1, 2, 3, 7, 64):
    r=_Reader(io.StringIO(" [  -12.34e+12, \"x\\n\" ] "),chunk_size=chunk)
    r.expect("[");assert r.value(",]")==-12.34e12;r.expect(",")
    assert r.value(",]")=="x\n";r.expect("]");assert not r.peek()

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);path=root/"large.json.gz";count=200000
    with gzip.open(path,"wt") as f:
        f.write("{\"h\":0.00001,\"traces\":[")
        for i in range(count):
            if i:f.write(",")
            f.write(json.dumps({"step":i,"accepted":True,"force":i/10},separators=(",",":")))
        f.write("],\"accepted_steps\":200000,\"rejected_steps\":0}")
    # Callback must see every step, including the last, without a trace list.
    seen=[0,0.0]
    def visit(metadata,row,index):
        assert row["step"]==index
        seen[0]+=1;seen[1]+=row["force"]
    tracemalloc.start();metadata,n=read_native(path,visit);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert n==count and seen[0]==count and metadata["accepted_steps"]==count
    assert "traces" not in metadata and peak<8*1024*1024,peak
    truncated=root/"truncated.json.gz";truncated.write_bytes(path.read_bytes()[:-12])
    try:read_native(truncated)
    except (EOFError,OSError,ValueError):pass
    else:raise AssertionError("truncated gzip was accepted")
    for payload in ("{\"traces\":[1,]}","{\"traces\":[],}","{\"traces\":[]}junk", "{\"traces\":[],\"traces\":[]}"):
        bad=root/"bad.json";bad.write_text(payload)
        try:read_native(bad)
        except ValueError:pass
        else:raise AssertionError("invalid JSON was accepted: "+payload)
print(f"streamed {count} rows; Python peak {peak} bytes; malformed/truncated inputs rejected")
