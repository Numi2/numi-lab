import unittest
import random
from unittest.mock import patch
from numilab_human import cardiac_cavity_intersections as ci


class IncrementalExactTests(unittest.TestCase):
    def records(self, crossing=True):
        z=-1 if crossing else 2
        x=[(0,0,0),(4,0,0),(0,4,0), (1,1,z),(1,1,3),(3,1,2),
           (20,20,0),(21,20,0),(20,21,0)]
        return ci._records(x,[(0,1,2),(3,4,5),(6,7,8)])

    def full(self,records):
        return ci._audit_self_pairs_touching_faces(records,{r[3] for r in records})

    def compare(self,baseline,records):
        updated,report=ci._update_complete_self_pair_baseline(baseline,records)
        full=self.full(records)
        self.assertEqual(report['triangle_pairs'],full['triangle_pairs'])
        self.assertEqual(report['count'],full['count'])
        self.assertTrue(report['complete_surface_audit'])
        return updated,report

    def test_new_and_resolved_crossings_match_full(self):
        b,r=ci._prepare_complete_self_pair_baseline(self.records(False))
        self.assertEqual(r['count'],0)
        b,r=self.compare(b,self.records(True))
        self.assertEqual(r['count'],1)
        self.assertEqual(r['selected_face_count'],1)
        b,r=self.compare(b,self.records(False))
        self.assertEqual(r['count'],0)

    def test_untouched_crossing_is_retained(self):
        records=self.records(True);b,_=ci._prepare_complete_self_pair_baseline(records)
        t=list(records[2][0]);t[2]=(20,22,0)
        moved=ci._records(list(t),[(0,1,2)])[0]
        moved=(moved[0],moved[1],moved[2],2,(6,7,8))
        _,r=self.compare(b,records[:2]+[moved])
        self.assertEqual(r['reused_unchanged_crossing_pairs'],1)

    def test_identical_geometry_skips_narrow_phase(self):
        records=self.records();b,_=ci._prepare_complete_self_pair_baseline(records)
        with patch.object(ci,'_audit_self_pairs_touching_faces',side_effect=AssertionError('unexpected audit')):
            _,r=ci._update_complete_self_pair_baseline(b,records[::-1])
        self.assertEqual(r['selected_face_count'],0)
        self.assertEqual(r['aabb_candidate_pairs'],0)
        self.assertEqual(r['count'],1)

    def test_added_removed_faces(self):
        records=self.records();b,_=ci._prepare_complete_self_pair_baseline(records[:1])
        b,r=self.compare(b,records)
        self.assertEqual(r['selected_face_count'],2)
        _,r=self.compare(b,records[1:])
        self.assertEqual(r['removed_face_count'],1)
        self.assertEqual(r['count'],0)

    def test_adjacency_identity_change_requires_recheck(self):
        points=[(0,0,0),(1,0,0),(0,1,0),(0,0,0),(-1,0,1),(0,-1,1)]
        records=ci._records(points,[(0,1,2),(3,4,5)])
        b,r=ci._prepare_complete_self_pair_baseline(records)
        self.assertEqual(r['count'],1)
        changed=ci._records(points,[(0,1,2),(0,4,5)])
        _,r=self.compare(b,changed)
        self.assertEqual(r['count'],0)
        self.assertEqual(r['selected_face_count'],1)

    def test_snapshot_is_deeply_immutable(self):
        records=self.records()
        mutable=[[list(map(list,r[0])),list(r[1]),list(r[2]),r[3],list(r[4])] for r in records]
        b,_=ci._prepare_complete_self_pair_baseline(mutable)
        mutable[1][0][0][0]=999
        _,r=self.compare(b,records)
        self.assertEqual(r['selected_face_count'],0)

    def test_rejects_stale_bounds_duplicate_ids_and_inconsistent_shared_coordinates(self):
        records=self.records()
        bad=list(records);r=bad[0];bad[0]=(r[0],(0,0,-1),r[2],r[3],r[4])
        with self.assertRaises(ci.HumanImportError):ci._prepare_complete_self_pair_baseline(bad)
        with self.assertRaises(ci.HumanImportError):ci._prepare_complete_self_pair_baseline(records+[records[0]])
        bad=list(records);r=bad[1];bad[1]=(*r[:4],(0,4,5))
        with self.assertRaises(ci.HumanImportError):ci._prepare_complete_self_pair_baseline(bad)

    def test_empty_surfaces(self):
        b,r=ci._prepare_complete_self_pair_baseline([])
        self.assertEqual(r['count'],0)
        b,r=self.compare(b,self.records())
        _,r=self.compare(b,[])
        self.assertEqual(r['count'],0)
        self.assertEqual(r['removed_face_count'],3)

    def test_repeated_local_updates_match_full_on_dense_pair_set(self):
        rng=random.Random(253)
        points=[]
        for face in range(12):
            while True:
                tri=[tuple(rng.randrange(-5,6) for _ in range(3)) for _ in range(3)]
                if any(ci._cross(ci._sub(tri[1],tri[0]),ci._sub(tri[2],tri[0]))):break
            points.extend(tri)
        faces=[(3*i,3*i+1,3*i+2) for i in range(12)]
        b,_=ci._prepare_complete_self_pair_baseline(ci._records(points,faces))
        for step in range(8):
            i=step%12
            for j in faces[i]:points[j]=tuple(v+(1 if axis==step%3 else 0) for axis,v in enumerate(points[j]))
            b,r=self.compare(b,ci._records(points,faces))
            self.assertEqual(r['selected_face_count'],1)


if __name__=='__main__':unittest.main()
