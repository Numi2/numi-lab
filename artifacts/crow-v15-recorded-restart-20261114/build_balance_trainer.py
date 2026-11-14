from pathlib import Path
import subprocess,hashlib,json
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';s=(r/'apps/task_train.swift').read_text();needle='physicsSubsteps: options.birdFlowJourneyVariant == .v14ObjectPush ? 32 : 4,';assert s.count(needle)==1
p=a/'task_train-balance-16.swift';p.write_text(s.replace(needle,'physicsSubsteps: options.birdFlowJourneyVariant == .v14ObjectPush ? 32 : 16,'))
out=b/'metalrobo_task_train_balance_16';cmd=['xcrun','swiftc','-O','-swift-version','6','-parse-as-library','-I',str(r/'include/metalrobo'),str(p),str(r/'apps/TrainingLifeCheckpoint.swift'),str(r/'apps/RunInspector.swift'),str(r/'bindings/swift/MetalRoboTaskRollout.swift'),'-L',str(b/'round-neural-host'),'-lmetalrobo','-framework','AppKit','-o',str(out)]
subprocess.run(cmd,check=True,cwd=r);(a/'balance-trainer-build.json').write_text(json.dumps({'command':cmd,'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p,out)}},indent=2)+'\n');print('built 16-substep trainer')
