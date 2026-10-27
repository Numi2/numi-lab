"""Bound a task-owned subprocess group; retain existing stage evidence."""
import argparse,os,signal,subprocess,sys,time,json
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--seconds",type=float,required=True)
 p.add_argument("argv",nargs=argparse.REMAINDER)
 a=p.parse_args();argv=a.argv[1:]if a.argv[:1]==["--"]else a.argv
 if not argv or not 1<=a.seconds<=1500:raise ValueError("bounded child command required")
 proc=subprocess.Popen(argv,start_new_session=True)
 try:code=proc.wait(timeout=a.seconds)
 except subprocess.TimeoutExpired:
  # The child PID remains reserved until wait() reaps it. Signal only its group.
  try:os.killpg(proc.pid,signal.SIGTERM)
  except ProcessLookupError:pass
  time.sleep(2)
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  proc.wait(timeout=5)
  print(json.dumps({"status":"METHOD_TIME_BUDGET_EXHAUSTED","seconds":a.seconds,"owned_process_group":proc.pid}),flush=True)
  return 124
 return code
if __name__=="__main__":sys.exit(main())
