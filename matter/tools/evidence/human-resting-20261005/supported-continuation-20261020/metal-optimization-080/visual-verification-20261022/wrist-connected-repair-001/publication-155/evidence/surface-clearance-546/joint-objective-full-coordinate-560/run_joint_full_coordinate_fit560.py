"""Full-coordinate construction adapter retaining every existing 458 goal row."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
from scipy import sparse

from joint_objective_560 import retain_joint_rows560
D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY = D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
ACTIVE_COMPONENT = D / "D/contained-clearance-545/component_lp_active545.py"
SURFACE_HELPER = D / "D/surface-clearance-546/surface_objective_546.py"
EXPECTED_ENTRY_SHA = "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"
EXPECTED_ACTIVE_SHA = "37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed"
EXPECTED_SURFACE_HELPER_SHA = "dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516"
EXPECTED_JOINT_HELPER_SHA = "09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(SURFACE_HELPER) != EXPECTED_SURFACE_HELPER_SHA:
    raise RuntimeError("pinned 546 outside-depth helper hash changed")
_surface_spec = importlib.util.spec_from_file_location("surface_objective_546_for_joint560", SURFACE_HELPER)
_surface_module = importlib.util.module_from_spec(_surface_spec)
_surface_spec.loader.exec_module(_surface_module)
select_outside_depth_rows546 = _surface_module.select_outside_depth_rows546


def main():
    helper_path = Path(__file__).with_name("joint_objective_560.py")
    frozen = {
        ENTRY: EXPECTED_ENTRY_SHA,
        ACTIVE_COMPONENT: EXPECTED_ACTIVE_SHA,
        SURFACE_HELPER: EXPECTED_SURFACE_HELPER_SHA,
    }
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise RuntimeError("a pinned 490/545/546 owner source changed")
    helper_sha = sha(helper_path)
    if helper_sha != EXPECTED_JOINT_HELPER_SHA:
        raise RuntimeError("joint objective helper hash changed")

    spec = importlib.util.spec_from_file_location("fit490_joint560", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original_load = entry._load_module
    receipt = {}

    def load490(path):
        runner = original_load(path)
        if Path(path).resolve() != entry.RUNNER_485.resolve():
            return runner
        original_runner_load = runner.load

        def load485(name, module_path):
            module = original_runner_load(name, module_path)
            if name == "component_lp_decomposition_485_for_run":
                spec_active = importlib.util.spec_from_file_location(
                    "component_active560", ACTIVE_COMPONENT
                )
                active = importlib.util.module_from_spec(spec_active)
                spec_active.loader.exec_module(active)
                module.solve_by_components = active.solve_by_components
                receipt["active_component_adapter_source_sha256"] = sha(ACTIVE_COMPONENT)
                return module
            if name != "matched458_for_485":
                return module
            install = module.install_objective

            def install_joint(owner, objective, crossing):
                result = install(owner, objective, crossing)
                basis = owner.build_immediate_parent_basis
                build = owner.build_clearance_rows
                depth = owner.build_output_node_direction_rows

                def full_basis(base, support, anchors):
                    B, parent, basis_report = basis(base, support, anchors)
                    scope = np.asarray(basis_report["output_scope_rows"], dtype=np.int64)
                    free = np.flatnonzero(~np.isin(scope, anchors))
                    Q = sparse.csr_matrix(
                        (np.ones(len(free)), (free, np.arange(len(free)))),
                        shape=(len(scope), len(free)),
                    )
                    basis_report.update(
                        control_parent_rows=scope[free].tolist(),
                        matrix_shape=list(Q.shape),
                        matrix_nnz=Q.nnz,
                        semantics=(
                            "independent emitted-node construction coordinates on the "
                            "existing expanded scope; source ancestry and native "
                            "mechanical coordinates unchanged"
                        ),
                        construction_full_output_coordinates=True,
                    )
                    receipt["construction_coordinate_nodes"] = int(len(free))
                    return Q, scope[free], basis_report

                def joint_rows(*args, **kwargs):
                    C, h, weights, metadata, proof = retain_joint_rows560(
                        build, *args, **kwargs
                    )
                    receipt.update(proof)
                    return C, h, weights, metadata

                def outside_rows(node_count, support, node_rows, normals, lower, typed_ids=None):
                    rows, dirs, lo, ids, proof = select_outside_depth_rows546(
                        node_rows, normals, lower, typed_ids
                    )
                    receipt.update(proof)
                    return depth(node_count, support, rows, dirs, lo, ids)

                owner.build_immediate_parent_basis = full_basis
                owner.build_clearance_rows = joint_rows
                owner.build_output_node_direction_rows = outside_rows
                return result

            module.install_objective = install_joint
            return module

        runner.load = load485
        return runner

    entry._load_module = load490
    self_sha = sha(__file__)
    entry.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    if sha(__file__) != self_sha or sha(helper_path) != helper_sha:
        raise RuntimeError("joint adapter changed during fit invocation")
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise RuntimeError("a pinned 490/545/546 owner changed during fit invocation")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["full_coordinate_joint560"] = {
        **receipt,
        "source_sha256": self_sha,
        "joint_objective_helper_sha256": helper_sha,
        "pinned_owner_sha256": {
            "matched_490_entry": EXPECTED_ENTRY_SHA,
            "545_component_adapter": EXPECTED_ACTIVE_SHA,
            "546_outside_depth_selector": EXPECTED_SURFACE_HELPER_SHA,
            "560_joint_row_adapter": EXPECTED_JOINT_HELPER_SHA,
        },
        "construction_only": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
