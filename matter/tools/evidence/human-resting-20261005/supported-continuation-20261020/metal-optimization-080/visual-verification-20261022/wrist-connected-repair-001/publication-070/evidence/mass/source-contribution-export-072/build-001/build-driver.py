#!/usr/bin/env python3
import json, pathlib, subprocess
B=pathlib.Path(__file__).resolve().parent
for name, logname in (("compile-command.json","compile-attempt-003.log"),("link-command.json","link.log")):
    cmd=json.loads((B/name).read_text())
    with (B/logname).open("wb") as out:
        p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
    if p.returncode: raise SystemExit(p.returncode)
