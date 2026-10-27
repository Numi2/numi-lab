from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("outward_boneface_filter640",HERE/"outward_boneface_filter640.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def axis(source,normal,*,index=0,sign=1,extra=1):
    return {"axis_source":source,"axis_index":index,"axis_sign":sign,"extra_sign":extra,
            "normal_world_unit":np.asarray(normal,dtype=np.float64)}

def tri_ccw():
    return np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])

def test_only_anti_outward_bone_face_axis_is_removed():
    axes=[axis("bone_face",[0,0,1],sign=1),axis("bone_face",[0,0,-1],sign=-1,extra=-1),
          axis("skin_face",[0,0,-1]),axis("edge_cross",[0,0,-1]),axis("edge_cross",[1,0,0])]
    kept,receipt=mod.filter_axis_inventory(axes,tri_ccw(),enabled=True)
    assert len(kept)==4 and all(got is want for got,want in zip(kept,[axes[0],axes[2],axes[3],axes[4]]))
    assert receipt["removed_axis_count"]==1 and receipt["opposed_bone_face_axis_count"]==1
    assert receipt["non_bone_face_axes_preserved"]

def test_disabled_mode_preserves_original_inventory():
    axes=[axis("bone_face",[0,0,1]),axis("bone_face",[0,0,-1],sign=-1,extra=-1),axis("edge_cross",[0,0,-1])]
    kept,receipt=mod.filter_axis_inventory(axes,tri_ccw(),enabled=False)
    assert len(kept)==len(axes) and all(got is want for got,want in zip(kept,axes))
    assert receipt["removed_axis_count"]==0 and receipt["output_axis_count"]==len(axes)

def test_filter_orientation_follows_triangle_winding():
    tri=tri_ccw()[::-1]
    axes=[axis("bone_face",[0,0,1]),axis("bone_face",[0,0,-1],sign=-1,extra=-1)]
    kept,receipt=mod.filter_axis_inventory(axes,tri,enabled=True)
    assert len(kept)==1 and kept[0] is axes[1]
    assert receipt["removed_axis_count"]==1

def test_nonparallel_bone_face_axis_fails_closed():
    axes=[axis("bone_face",[1,0,0])]
    with pytest.raises(ValueError,match="does not match"):
        mod.filter_axis_inventory(axes,tri_ccw(),enabled=True)

def test_degenerate_triangle_fails_closed():
    axes=[axis("edge_cross",[1,0,0])]
    tri=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    with pytest.raises(ValueError,match="degenerate"):
        mod.filter_axis_inventory(axes,tri,enabled=True)
