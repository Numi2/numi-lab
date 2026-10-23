"""Bind compact exact-audit face rows to anatomical mesh identities.

The geometry engine keeps face IDs in the scene passed to it. Bone scenes
contain a compact selected-skin prefix followed by bone faces; a local skin
row is never an index into the complete skin mesh.
"""
from copy import deepcopy

def bind_skin_witness_identities(check, faces, *, patch_rows=None,
                                 bone_owners=None, face_origins=None):
    result=deepcopy(check)
    if (patch_rows is None)!=(bone_owners is None):
        raise ValueError("compact skin/bone scene needs both row and owner maps")
    n=len(faces)
    def skin(face):
        if face<0 or face>=n:raise ValueError("skin witness exceeds global face domain")
        row={"skin_face":int(face),"skin_vertices":[int(v) for v in faces[face]]}
        if len(row["skin_vertices"])!=3:raise ValueError("skin witness is not triangular")
        if face_origins is not None:
            if len(face_origins)!=n:raise ValueError("face origin map has wrong size")
            row["face_origin_code"]=int(face_origins[face])
        return row
    for witnesses in result.get("witnesses_by_owner_pair",{}).values():
        for w in witnesses:
            a,b=int(w["face_a"]),int(w["face_b"])
            if patch_rows is None:
                w["anatomical_face_a"]=skin(a)
                w["anatomical_face_b"]=skin(b)
            else:
                if a<0 or a>=len(patch_rows):raise ValueError("bone-check skin witness is not a compact patch row")
                bone=b-len(patch_rows)
                if bone<0 or bone>=len(bone_owners):raise ValueError("bone-check witness exceeds bone domain")
                w["anatomical_face_a"]=skin(int(patch_rows[a]))
                w["anatomical_face_b"]={"bone_face":bone,"bone_owner":str(bone_owners[bone])}
    result["witness_identity_domains"]={"face_a":"compact_skin_patch_row" if patch_rows is not None else "global_skin_face", "face_b":"skin_prefix_then_bone_row" if patch_rows is not None else "global_skin_face", "anatomical_face_a":"explicit_global_skin_identity", "anatomical_face_b":"explicit_bone_identity" if patch_rows is not None else "explicit_global_skin_identity"}
    return result
