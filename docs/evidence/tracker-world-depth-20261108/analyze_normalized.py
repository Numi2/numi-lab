#!/usr/bin/env python3
import json,re,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); out=Path(sys.argv[2]); failures=[]
def check(cond,label):
    if not cond:failures.append(label)
def loginfo(name):
    log=(root/(name+'.log')).read_text()
    mt=re.search(r'depth_width=(\d+) depth_height=(\d+) depth_features=(\d+)',log)
    mw=re.search(r'output_per_env=(\d+)',log)
    trans=re.findall(r'transition\[(\d+)\] done=(\d+) timeout=(\d+) physics_error=(\d+) reason=(\d+) reward=([^\n]+)',log)
    if not mt or not mw or len(trans)!=60:raise ValueError('incomplete log '+name)
    return tuple(map(int,mt.groups())),int(mw.group(1)),trans
try:
    meta,width,trans=loginfo('first')
    for name in ('repeat','control'):
        mi,wi,ti=loginfo(name);check((mi,wi,ti)==(meta,width,trans),'log mismatch '+name)
    w,h,features=meta;pix=w*h*2
    check(width==102+pix+features,'layout width')
    for ext in ('q','v','scene','actor','status','reset','history','observation'):
        check((root/('first.'+ext)).read_bytes()==(root/('repeat.'+ext)).read_bytes(),'repeat '+ext)
    for ext in ('q','v','scene','actor','status','reset'):
        check((root/('first.'+ext)).read_bytes()==(root/('control.'+ext)).read_bytes(),'control '+ext)
    data=np.fromfile(root/'first.observation',dtype='<u4')
    check(data.size==30*3*2*width,'capture size')
    rows=data.reshape(30,3,2,width)
    reset=np.fromfile(root/'first.reset',dtype='<u4').reshape(30,2)
    failed=np.array([int(t[3])!=0 for t in trans]).reshape(30,2)
    visual_tracks={h*34+i for h in range(3) for i in range(7)}
    counts={k:0 for k in ('accepted','rejected_reset','rejected_nonreset','accepted_depth_changed','rejected_depth_changed','mixed','accepted_reset','accepted_visible_depth','rejected_visible_depth')}
    far=np.float32(1.0).view(np.uint32)
    for step in range(30):
        counts['mixed']+=int(failed[step,0]!=failed[step,1])
        for env in range(2):
            pre,mid,post=rows[step,:,env,:];fail=failed[step,env];rst=bool(reset[step,env])
            visible=bool(np.any(mid[102:102+pix].view('<f4')<1.0))
            counts['rejected_visible_depth' if fail else 'accepted_visible_depth']+=int(visible)
            if not fail:
                counts['accepted']+=1;counts['accepted_reset']+=int(rst)
                counts['accepted_depth_changed']+=int(not np.array_equal(pre[102:],mid[102:]))
                check(np.array_equal(post,mid),f'accepted {step}/{env}')
            elif not rst:
                counts['rejected_nonreset']+=1
                counts['rejected_depth_changed']+=int(not np.array_equal(pre[102:],mid[102:]))
                check(np.array_equal(post,pre),f'rejected {step}/{env}')
            else:
                counts['rejected_reset']+=1
                counts['rejected_depth_changed']+=int(not np.array_equal(pre[102:],mid[102:]))
                mask=np.ones(102,dtype=bool);mask[list(visual_tracks)]=False
                check(np.array_equal(post[:102][mask],pre[:102][mask]),f'reset nonvisual {step}/{env}')
                check(np.all(post[:102][list(visual_tracks)]==0),f'reset tracks {step}/{env}')
                check(np.all(post[102:102+pix]==far),f'reset depth {step}/{env}')
                check(np.all(post[102+pix:]==0),f'reset features {step}/{env}')
    for k in ('accepted','rejected_reset','rejected_nonreset','accepted_depth_changed','rejected_depth_changed','mixed'):
        check(counts[k]>0,'vacuous '+k)
    report=dict(layout=dict(width=w,height=h,features=features,output=width),counts=counts,failures=failures,passed=not failures)
except Exception as e:
    report=dict(passed=False,failures=failures+[type(e).__name__+': '+str(e)])
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
