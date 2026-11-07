import hashlib,json,sys
p=sys.argv[1]
m=json.load(open(p))
print(json.dumps({"schema":"numi.science.prediction.v1","model_sha256":hashlib.sha256(open(p,"rb").read()).hexdigest(),"prediction":m["prediction"]}))
