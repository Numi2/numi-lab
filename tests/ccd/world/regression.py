#!/usr/bin/env python3
import json,subprocess,sys,os
from pathlib import Path
exe,shader,output=map(Path,sys.argv[1:]);output.mkdir(parents=True,exist_ok=True)
with (output/"world-steps.json").open("w") as out,(output/"world-steps.stderr").open("w") as err:
 subprocess.run([str(exe),str(shader)],stdout=out,stderr=err,check=True,env=dict(os.environ,NUMI_CCD_EVENT_TRACE="1"))
with (output/"analysis.json").open("w") as out:
 subprocess.run([sys.executable,str(Path(__file__).with_name("analyze.py")),str(output/"world-steps.json")],stdout=out,check=True)
