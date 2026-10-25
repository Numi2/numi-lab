import csv, hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER = D / "connected-local-witness-proposal-235/emit_empty_point_targets_235.py"
BASE = D / "D/connected-local-witness-proposal-222/propose_joint_clearance_witnesses_222.py"

def loadmod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

class EmptyPointTargetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.emitter = loadmod(OWNER, "empty_point_emitter_235")
        self.loader = loadmod(BASE, "clearance_loader_222")
        self.candidate = self.root / "candidate.npz"
        self.query = self.root / "query"
        self.query.mkdir()
        self.points = np.asarray([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        self.faces = np.asarray([[0,1,2]], dtype=np.int64)
        with self.candidate.open("wb") as stream:
            np.savez_compressed(stream, points_world_m=self.points, source_O_faces=self.faces)
        self._write_query()
        self.summary_path = self.root / "summary.json"
        self._write_summary()
        self.point_npz = self.root / "empty-point-targets.npz"
        self.point_report = self.root / "empty-point-targets.json"
        self.face_npz = self.root / "face-targets.npz"
        self.face_report = self.root / "face-targets.json"
        with self.face_npz.open("wb") as stream:
            np.savez_compressed(stream, face_rows=np.empty(0,np.int64),
                face_node_rows=np.empty((0,3),np.int64),
                barycentric_weights=np.empty((0,3),np.float64),
                inside_sample_world_m=np.empty((0,3),np.float64),
                target_world_f32_m=np.empty((0,3),np.float32),
                nearest_boundary_direction_world_unit=np.empty((0,3),np.float64),
                target_exactly_outside_all185=np.empty(0,np.bool_))
        self.face_report.write_text(json.dumps({"inputs":{"candidate_sha256":sha(self.candidate)},
            "passing_targets":0,"result_rows":0,"candidate_points":3,"candidate_faces":1,
            "failure_counts":{},"matched_159_complete":True}))

    def tearDown(self):
        self.tmp.cleanup()

    def _write_query(self, inside=0, ambiguous=0, invalid=0, bad_banks=False):
        p_fields=["query_row","x_bits_u32","y_bits_u32","z_bits_u32"]
        p_rows=[]
        for i,point in enumerate(self.points):
            bits=point.view(np.uint32)
            p_rows.append(dict(query_row=i,x_bits_u32=int(bits[0]),y_bits_u32=int(bits[1]),z_bits_u32=int(bits[2])))
        write_csv(self.query/"points.csv",p_fields,p_rows)
        n_fields=["query_row","total_bone_banks","aabb_proven_outside_banks","parity_outside_banks",
                  "inside_banks","ambiguous_banks","invalid_banks"]
        n_rows=[]
        for i in range(3):
            n_rows.append(dict(query_row=i,total_bone_banks=185,
                aabb_proven_outside_banks=(184 if bad_banks else 185-inside-ambiguous-invalid),
                parity_outside_banks=0,inside_banks=inside,ambiguous_banks=ambiguous,invalid_banks=invalid))
        write_csv(self.query/"nodes.csv",n_fields,n_rows)
        pair_fields=["query_row","bank_body","bank_stable_id","classification","ambiguous"]
        pair_rows=[dict(query_row=0,bank_body=10,bank_stable_id=3,classification="outside",ambiguous=0)]
        if inside:
            pair_rows=[dict(query_row=0,bank_body=11,bank_stable_id=5,classification="inside",ambiguous=0)]
        if ambiguous:
            pair_rows.append(dict(query_row=0,bank_body=12,bank_stable_id=6,classification="ambiguous",ambiguous=1))
        write_csv(self.query/"pairs.csv",pair_fields,pair_rows)
        receipt={"query_rows":3,"identity_domain":"mechanics_node_row","physical_steps":0,
            "points_sha256":sha(self.query/"points.csv"),"pairs_sha256":sha(self.query/"pairs.csv"),
            "nodes_sha256":sha(self.query/"nodes.csv")}
        (self.query/"query-receipt.json").write_text(json.dumps(receipt))

    def _write_summary(self, inside_nodes=0, inside_pairs=0, query_pairs=1, wrong_hash=False):
        self.summary_path.write_text(json.dumps({"candidate_sha256":("0"*64 if wrong_hash else sha(self.candidate)),
            "points":3,"faces":1,"accepted_physical_steps":0,"geometry_admitted":False,
            "complete_surface_audit":True,"exact_PLC_surface_pairs":0,
            "inside_nodes":inside_nodes,"inside_node_bank_pairs":inside_pairs,
            "query_pair_count":query_pairs,"inside_witnesses":([] if inside_pairs==0 else [{"query_row":0}])}))

    def test_zero_row_emission_is_accepted_by_frozen_222_loader(self):
        result=self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
                                                    self.point_npz,self.point_report)
        self.assertEqual(result["targets"],0)
        with np.load(self.point_npz,allow_pickle=False) as z:
            expected={"query_rows":((0,),np.dtype("int64")),
                "target_world_f32_m":((0,3),np.dtype("float32")),
                "proposal_direction_world_unit":((0,3),np.dtype("float64")),
                "source_surface_normal_world_unit":((0,3),np.dtype("float64")),
                "source_normal_available_mask":((0,),np.dtype("bool")),
                "pass_mask":((0,),np.dtype("bool"))}
            for key,(shape,dtype) in expected.items():
                self.assertEqual(z[key].shape,shape)
                self.assertEqual(z[key].dtype,dtype)
        loaded,_=self.loader.load_clearance_target_inputs(
            self.point_npz,self.point_report,self.face_npz,self.face_report,
            self.candidate,self.points,self.faces)
        self.assertEqual(len(loaded["point_rows"]),0)
        report=json.loads(self.point_report.read_text())
        self.assertEqual(report["status"],"PASS_FIRST_UNION_EXIT_TARGETS_ONLY")
        self.assertEqual(report["physical_steps"],0)
        self.assertEqual(report["targets"],[])

    def test_inside_query_rejected(self):
        self._write_query(inside=1)
        self._write_summary(inside_nodes=1,inside_pairs=1,query_pairs=1)
        with self.assertRaisesRegex(ValueError,"inside_nodes"):
            self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
                self.root/"bad.npz",self.root/"bad.json")

    def test_ambiguous_bank_rejected(self):
        self._write_query(ambiguous=1)
        self._write_summary(query_pairs=1)
        with self.assertRaisesRegex(ValueError,"not strictly outside"):
            self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
                self.root/"bad.npz",self.root/"bad.json")

    def test_incomplete_185_bank_accounting_rejected(self):
        self._write_query(bad_banks=True)
        self._write_summary()
        with self.assertRaisesRegex(ValueError,"incomplete bank accounting"):
            self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
                self.root/"bad.npz",self.root/"bad.json")

    def test_candidate_hash_mismatch_rejected(self):
        self._write_summary(wrong_hash=True)
        with self.assertRaisesRegex(ValueError,"candidate hash"):
            self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
                self.root/"bad.npz",self.root/"bad.json")

    def test_nonzero_pair_count_in_empty_report_rejected_by_222_loader(self):
        self.emitter.emit_empty_point_targets(self.candidate,self.summary_path,self.query,
            self.point_npz,self.point_report)
        report=json.loads(self.point_report.read_text())
        report["inside_node_bank_pairs"]=1
        self.point_report.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,"point target report"):
            self.loader.load_clearance_target_inputs(self.point_npz,self.point_report,
                self.face_npz,self.face_report,self.candidate,self.points,self.faces)

if __name__=="__main__":
    unittest.main()
