from pathlib import Path
import json,os,sys
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root/'python'));os.environ['METALROBO_LIBRARY']=str(root/'lib/libmetalrobo.dylib')
from metalrobo.crow_retinal_normalization import calibrate_initial_policy
out=root/'calibration';out.mkdir();result=calibrate_initial_policy(root/'input/calibration-source.policypack',root/'input/calibration.rolloutpack',out)
print(json.dumps(result))
