from pathlib import Path
import importlib.util
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("prepare541",HERE/"prepare_face_ancestry_541.py")
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
CANDIDATE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/focused-clearance-541/topology/run-003-filtered/candidate.npz")
def test_real_541_surface_ancestry():
    with np.load(CANDIDATE,allow_pickle=False) as z:
        report=M.validate_arrays({k:z[k] for k in z.files})
    assert report["face_count"]==41733
    assert report["point_count"]==20889
    assert report["face_source_ancestry"]["multi_parent_row_count"]>=1
    assert report["immediate_parent_face_ancestry"]["multi_parent_row_count"]>=1
    assert report["multi_parent_single_label_sentinel_count"]==report["immediate_parent_face_ancestry"]["multi_parent_row_count"]
    assert report["layers_or_volume_cells_present"] is False
    assert report["material_or_mass_weights_present"] is False
def test_reject_unsorted_parent_csr():
    with np.load(CANDIDATE,allow_pickle=False) as z: q={k:z[k] for k in z.files}
    x=q["face_source_ancestry_face_rows"].copy()
    widths=np.diff(q["face_source_ancestry_indptr"])
    row=int(np.flatnonzero(widths>1)[0])
    a,b=q["face_source_ancestry_indptr"][row:row+2]
    assert b-a>1
    x[a:b]=x[a:b][::-1]
    q["face_source_ancestry_face_rows"]=x
    with pytest.raises(ValueError,match="unsorted"):
        M.validate_arrays(q)
def test_reject_collapsed_multisource_label():
    with np.load(CANDIDATE,allow_pickle=False) as z: q={k:z[k] for k in z.files}
    widths=np.diff(q["refinement_face_parent_ancestry_indptr"])
    row=int(np.flatnonzero(widths>1)[0])
    q["refinement_face_parent_face_row"]=q["refinement_face_parent_face_row"].copy()
    q["refinement_face_parent_face_row"][row]=int(q["refinement_face_parent_ancestry_face_rows"][q["refinement_face_parent_ancestry_indptr"][row]])
    with pytest.raises(ValueError,match="multi-parent faces"):
        M.validate_arrays(q)
