from human_skin_witness_identity import bind_skin_witness_identities as bind
import copy
faces=[[i,i+1,i+2] for i in range(13)];origins=[0]*13;origins[7]=2;origins[11]=1
base={"intersection_pairs":1,"witnesses_by_owner_pair":{"skin|humerus_l":[{"face_a":1,"face_b":3,"intersection_points":2}]}}
old=copy.deepcopy(base)
out=bind(base,faces,patch_rows=[7,11],bone_owners=["radius_l","humerus_l"],face_origins=origins)
w=out["witnesses_by_owner_pair"]["skin|humerus_l"][0]
assert w["anatomical_face_a"]=={"skin_face":11,"skin_vertices":[11,12,13],"face_origin_code":1}
assert w["anatomical_face_b"]=={"bone_face":1,"bone_owner":"humerus_l"}
assert base==old
out=bind(base,faces,face_origins=origins);assert out["witnesses_by_owner_pair"]["skin|humerus_l"][0]["anatomical_face_a"]["skin_face"]==1
for a,b in [(-1,2),(2,2),(0,1),(0,4)]:
    x=copy.deepcopy(base);x["witnesses_by_owner_pair"]["skin|humerus_l"][0].update(face_a=a,face_b=b)
    try:bind(x,faces,patch_rows=[7,11],bone_owners=["radius_l","humerus_l"])
    except ValueError:pass
    else:raise AssertionError((a,b))
print("witness identity checks passed: noncontiguous compact map, global map, immutability, four invalid domains")
