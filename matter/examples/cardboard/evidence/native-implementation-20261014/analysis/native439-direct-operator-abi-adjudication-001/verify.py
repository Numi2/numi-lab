#!/usr/bin/env python3
"""Recheck immutable native439 files and all source/runtime/external inputs."""
import hashlib, json
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()
def main():
    pkg=Path(__file__).resolve().parent
    report=json.loads((pkg/"report.json").read_text())
    base=Path(report["suite_path"])
    manifest=json.loads((base/"manifest.json").read_text())["files"]
    for rel,meta in manifest.items():
        b=(base/rel).read_bytes()
        assert len(b)==meta["bytes"] and sha(b)==meta["sha256"], rel
    assert sha((base/"manifest.json").read_bytes())==report["suite_manifest_sha256"]
    assert sha((base/"receipt.json").read_bytes())==report["suite_receipt_sha256"]
    assert len(report["cases"])==5
    for c in report["cases"]:
        case=base/c["case"]
        r=json.loads((case/"receipt.json").read_text())
        pred=json.loads((case/"prediction.json").read_text())
        assert r["execution_started"] and r["exit_code"]==0 and not r["timed_out"]
        assert r["source_inputs_unchanged"] and r["changed_inputs"]==[]
        assert pred["executed_build_verified"]
        ext={Path(x).name:Path(x) for x in pred["external_inputs"]}
        assert len(pred["inputs_sha256"])==797
        for key,expected in pred["inputs_sha256"].items():
            if key.startswith("source/"):
                p=Path(pred["source_snapshot"])/key[len("source/"):]
            elif key.startswith("runtime/"):
                p=Path(pred["runtime_dir"])/key[len("runtime/"):]
            elif key.startswith("external/"):
                p=ext.get(key.split("/",1)[1])
            else:
                raise AssertionError("unknown input binding: "+key)
            assert p is not None and p.is_file() and sha(p.read_bytes())==expected, key
        assert sha((case/"stdout.log").read_bytes())==c["stdout"]["sha256"]
        assert sha((case/"stderr.log").read_bytes())==c["stderr"]["sha256"]
        assert sha(Path(c["tool_path"]).read_bytes())==c["tool_sha256"]
    print("PASS: suite files and all five receipts plus 3,985 recorded input hashes revalidated")
    return 0
if __name__=="__main__": raise SystemExit(main())
