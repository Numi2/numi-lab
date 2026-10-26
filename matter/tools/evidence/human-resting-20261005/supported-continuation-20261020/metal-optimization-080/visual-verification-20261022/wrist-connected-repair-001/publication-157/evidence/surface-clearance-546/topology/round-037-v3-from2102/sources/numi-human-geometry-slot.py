#!/usr/bin/env python3
"""Serialize actual substantial Numi Human CPU/GPU jobs, never authoring time."""
import fcntl,os,sys,time
if len(sys.argv)<2:
    raise SystemExit("usage: numi-human-geometry-slot.py executable [args...]")
fd=os.open("/tmp/numi-human-connected-repair-heavy.lock",os.O_RDWR|os.O_CREAT,0o600)
print("waiting_for_human_geometry_slot pid="+str(os.getpid()),flush=True)
fcntl.flock(fd,fcntl.LOCK_EX)
os.ftruncate(fd,0)
os.write(fd,("pid="+str(os.getpid())+" started_unix="+str(time.time())+" command="+repr(sys.argv[1:])+"\n").encode())
os.set_inheritable(fd,True)
print("acquired_human_geometry_slot pid="+str(os.getpid()),flush=True)
os.execvp(sys.argv[1],sys.argv[1:])

