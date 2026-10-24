
"""Regression for exact posed seam identities; no physical stepping."""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from numilab_human import cardiac_cavity_intersections as audit

class SeamRecordsTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [(8.,8.,8.),(1.,0.,0.),(0.,1.,0.),
                         (0.,0.,1.),(0.,0.,0.),(0.,0.,0.)]
        self.faces = [(0,1,2),(4,1,2),(5,2,3)]

    def test_group_zero_cannot_alias_original_vertex_zero(self):
        rec, ids = audit.float32_surface_records(self.vertices,self.faces,
                                                source_seam_groups=[(4,5)])
        self.assertEqual(ids, (0,1,2,3,6,6))
        self.assertNotEqual(ids[0],ids[4])
        for face, row in zip(self.faces,rec):
            self.assertEqual(row[0],tuple(audit.float32_point_lattice_key(self.vertices[v]) for v in face))
        self.assertEqual(rec[0][4],(0,1,2))
        self.assertEqual(rec[1][4],(6,1,2))
        self.assertEqual([r[3] for r in rec],[0,1,2])

    def test_diverged_proven_group_keeps_original_vertices(self):
        self.vertices[5] = (0.,0.,0.125)
        rec,ids = audit.float32_surface_records(self.vertices,self.faces,
                                               source_seam_groups=[(4,5)])
        self.assertEqual(ids,tuple(range(6)))
        self.assertEqual(rec,audit._records([audit.float32_point_lattice_key(p) for p in self.vertices],self.faces))

    def test_coincidence_without_source_proof_not_shared(self):
        _,ids = audit.float32_surface_records(self.vertices,self.faces)
        self.assertNotEqual(ids[4],ids[5])

    def test_proven_seam_shared_feature_has_no_false_crossing(self):
        rec,_ = audit.float32_surface_records(self.vertices,self.faces[1:],
                                              source_seam_groups=[(4,5)])
        self.assertEqual(audit._audit_pair(rec,rec,same_surface=True)["count"],0)

    def test_shared_edge_does_not_hide_area_overlap(self):
        verts=[(0.,0.,0.),(2.,0.,0.),(0.,2.,0.),
               (0.,0.,0.),(2.,0.,0.),(1.,1.,0.)]
        rec,_=audit.float32_surface_records(verts,[(0,1,2),(3,4,5)],
                                            source_seam_groups=[(0,3),(1,4)])
        self.assertEqual(audit._audit_pair(rec,rec,same_surface=True)["count"],1)

    def test_reject_invalid_groups(self):
        for groups in [[(4,)],[(4,4)],[(4,5),(5,1)],[(-1,4)],[(4,6)],[(4.,5)],[(True,4)]]:
            with self.subTest(groups=groups),self.assertRaises(audit.HumanImportError):
                audit.float32_surface_records(self.vertices,self.faces,source_seam_groups=groups)

    def test_reject_invalid_faces_and_non_float32_geometry(self):
        for faces in [[(-1,1,2)],[(0,1,6)],[(0.,1,2)],[(True,1,2)],[(0,1)]]:
            with self.subTest(faces=faces),self.assertRaises(audit.HumanImportError):
                audit.float32_surface_records(self.vertices,faces)
        for value in [0.1,float("nan"),float("inf"),1e100]:
            vertices=list(self.vertices);vertices[0]=(value,0.,0.)
            with self.subTest(value=value),self.assertRaises(audit.HumanImportError):
                audit.float32_surface_records(vertices,self.faces)
    def test_empty_and_generator_groups(self):
        self.assertEqual(audit.float32_surface_records([],[]),([],()))
        rec,ids=audit.float32_surface_records(self.vertices,self.faces,
                            source_seam_groups=((x for x in (4,5)),))
        self.assertEqual(ids[4],ids[5])
if __name__=="__main__":unittest.main()
