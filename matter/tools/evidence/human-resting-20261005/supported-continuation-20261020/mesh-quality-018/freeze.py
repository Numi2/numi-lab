#!/usr/bin/env python3
import pathlib,re,csv,json,hashlib,subprocess
ROOT=pathlib.Path.cwd();OUT=pathlib.Path(__file__).resolve().parent
p=ROOT/"apps/NumiHumanWholeBodySkinReduction.hpp";h=subprocess.check_output(["git","show","c927871bafc1194f8ef314fc0687e4d123d11dd8:apps/NumiHumanWholeBodySkinReduction.hpp"],text=True);before=hashlib.sha256(h.encode()).hexdigest()
assert before=="a0e6d69706611257556c74ab7815ca2605f1f3c5f5530d9dd3472ecf482f5200"
rows=list(csv.DictReader((OUT/"candidate-layers.csv").open()));n=len(rows)
geom=json.loads((OUT/"float32-geometry.json").read_text());quality=json.loads((OUT/"quality.json").read_text())
assert geom["overall_float32_geometry_ok"]==1 and quality["outer_unchanged"]
receipt=dict(scope="Bounded inferred inner-layer mesh-quality correction; exterior, source identities, topology and ancestry preserved; geometry admission only, native evolution unqualified.",
 previous_header_sha256=before,previous_layer_receipt_sha256="f99da232e900ad80bad3dc13f4dc9e1479aa1b9b81d7e285d98c97aef769d7bb",
 coordinates_sha256=hashlib.sha256((OUT/"candidate-layers.csv").read_bytes()).hexdigest(),
 quality=quality,float32_geometry=geom,
 unchanged_mass_debit_kg=dict(skin=.0034525707761420958*1000,fat=.011652343293850826*900))
(OUT/"layer-geometry-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
def arr3(name,values):
 lines=[f"inline constexpr std::array<std::array<double,3>,{n}> {name}={{{{"]
 for i in range(0,n,3):
  lines.append("    "+", ".join("std::array<double,3>{"+",".join(format(float(x),".17g") for x in v)+"}" for v in values[i:i+3])+",")
 return "\n".join(lines+["}};"])
def arr1(name,values):
 lines=[f"inline constexpr std::array<double,{n}> {name}={{"]
 for i in range(0,n,10):lines.append("    "+", ".join(format(float(x),".17g") for x in values[i:i+10])+",")
 return "\n".join(lines+["};"])
for name,prefix in (("kWholeBodySkinInnerWorld","skin"),("kWholeBodySkinInnerLocal","skin_local"),("kWholeBodyFatDeepWorld","fat"),("kWholeBodyFatDeepLocal","fat_local")):
 repl=arr3(name,[[r[prefix+"_"+a] for a in "xyz"] for r in rows])
 h,count=re.subn(r"inline constexpr [^\n]* "+name+r"=.*?;\n",lambda m:repl+"\n",h,flags=re.S);assert count==1,name
for name,col in (("kWholeBodySkinThicknessM","skin_m"),("kWholeBodyFatThicknessM","fat_m")):
 repl=arr1(name,[r[col] for r in rows])
 h,count=re.subn(r"inline constexpr [^\n]* "+name+r"=.*?;\n",lambda m:repl+"\n",h,flags=re.S);assert count==1,name
for name,value in (("kWholeBodySkinLayerVolumeM3",quality["candidate_volumes_m3"][0]),("kWholeBodyFatLayerVolumeM3",quality["candidate_volumes_m3"][1])):
 h,count=re.subn(r"(inline constexpr double "+name+r"=)[^;]+;",lambda m:m[1]+format(value,".17g")+";",h);assert count==1
for name,value in (("kWholeBodySkinLayerCoordinatesSHA256",receipt["coordinates_sha256"]),("kWholeBodySkinLayerReceiptSHA256",hashlib.sha256((OUT/"layer-geometry-receipt.json").read_bytes()).hexdigest()),("kWholeBodySkinLayerFitMethod","bounded inner-boundary quality correction; exact Float32 geometry and original exterior/topology/ancestry retained; original donor mass debits preserved; native evolution unqualified")):
 h,count=re.subn(r"(inline constexpr std::string_view "+name+r'=)"[^"]+";',lambda m:m[1]+json.dumps(value)+";",h);assert count==1
h=h.replace("inline constexpr double kWholeBodySkinLayerVolumeM3=",
 "// Retain the previous registered donor mass partition through inferred layer-volume repairs.\ninline constexpr double kWholeBodySkinMassDebitKg=3.4525707761420958;\ninline constexpr double kWholeBodyFatMassDebitKg=10.487108964465744;\ninline constexpr double kWholeBodySkinLayerVolumeM3=",1)
h=h.replace("inline constexpr char kWholeBodySkinSourcePayloadSHA256[]",'// Inferred deep-fat attachments selected on the retained c927871b reference.\n// Freeze anatomical source/donor identities so inner-boundary repairs cannot\n// silently change mechanical constraints. These are not measured entheses.\nstruct WholeBodySkinDeepAttachmentSite {\n    std::uint32_t reducedVertex,sourceVertex,donorBody;\n};\ninline constexpr std::array<WholeBodySkinDeepAttachmentSite,20> kWholeBodySkinDeepAttachmentSites={{\n    {790u,13555u,20u},\n    {1333u,29464u,23u},\n    {656u,6723u,49u},\n    {2267u,52504u,99u},\n    {804u,14506u,139u},\n    {1719u,37264u,152u},\n    {1790u,39356u,153u},\n    {1557u,33621u,20u},\n    {1478u,32578u,23u},\n    {373u,4474u,49u},\n    {2166u,49004u,99u},\n    {752u,11704u,139u},\n    {1842u,41702u,152u},\n    {1831u,40968u,153u},\n    {1591u,34352u,20u},\n    {1104u,21455u,23u},\n    {297u,1569u,49u},\n    {2281u,52937u,99u},\n    {692u,8467u,139u},\n    {1987u,46481u,153u},\n}};\n'+"inline constexpr char kWholeBodySkinSourcePayloadSHA256[]",1)
p.write_text(h)
print("header",before,"->",hashlib.sha256(h.encode()).hexdigest())

