from pathlib import Path
import difflib
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
BASE = B / 'analysis/cardboard-effective-solver-environment-source161-002/source'
O = B / 'analysis/diagnostic-checkpoint-prefix-and-portable-tests-001'
assert not O.exists()
O.mkdir()
FILES = ['cardboard_solver_environment.py', 'cardboard_source_bound_run.py',
         'test_cardboard_solver_environment.py', 'test_cardboard_resolution_study.py']
texts = {name:(BASE / 'matter/tools' / name).read_text() for name in FILES}

helper = '''
def _pending_checkpoint_capture(case: Path, key: str,
                                native_argv: Sequence[str], native_exit_code: int | None,
                                effective_contract: Mapping[str, object] | None) -> dict[str, object]:
    """Prove a requested capture is strictly after a saved accepted prefix.

    Only the two explicit selectors used by coupon continuation are supported.
    A normal completed/failed run or an already-visited target cannot use this
    exception. The returned status is diagnostic-only, never a missing-capture
    success or a completed loading history.
    """
    if native_exit_code != 0 or effective_contract is None:
        raise ValueError("missing diagnostic capture has no successful accepted checkpoint boundary")
    validate_contract(effective_contract, allow_diagnostic=True)
    if effective_contract.get("schema") != DIAGNOSTIC_CONTRACT_SCHEMA:
        raise ValueError("pending checkpoint capture requires a diagnostic contract")
    flag = "--coupon-checkpoint-after-control"
    indices = [i for i, value in enumerate(native_argv) if value == flag]
    if len(indices) != 1 or indices[0]+1 >= len(native_argv):
        raise ValueError("missing diagnostic capture has no explicit coupon checkpoint request")
    raw_control = native_argv[indices[0]+1]
    if not re.fullmatch(r"[0-9]+", raw_control) or int(raw_control) >= UINT32_MAX:
        raise ValueError("invalid coupon checkpoint control")
    control = int(raw_control)
    output = case / "output"
    if output.is_symlink() or not output.is_dir() or output.resolve(strict=True) != output:
        raise ValueError("coupon checkpoint output directory is aliased")
    def regular_file(relative: str, cap: int) -> Path:
        rel = Path(relative)
        path = output / rel
        if (rel.is_absolute() or not rel.parts or ".." in rel.parts or path.is_symlink()
                or not path.is_file() or path.resolve(strict=True) != path):
            raise ValueError("coupon checkpoint reference escaped its output directory")
        stat = path.stat()
        if stat.st_nlink != 1 or not 0 < stat.st_size <= cap:
            raise ValueError("coupon checkpoint reference is not a bounded regular file")
        return path
    def digest(path: Path) -> str:
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                result.update(block)
        return result.hexdigest()
    result_path = regular_file("result.json", 64*1024)
    prefix_path = regular_file("coupon-checkpoint-prefix.json", 64*1024)
    result, prefix = (json.loads(p.read_text()) for p in (result_path, prefix_path))
    if (result.get("schema") != "numi.cardboard.probe-result.v1"
            or result.get("status") != "accepted_checkpoint_prefix_saved"
            or type(result.get("accepted_steps")) is not int
            or result["accepted_steps"] != control+1 or result.get("physical_validation") is not False
            or prefix.get("schema") != "numi.cardboard.coupon-checkpoint-prefix.v1"
            or prefix.get("status") != "accepted_checkpoint_prefix_saved"
            or type(prefix.get("accepted_control_index")) is not int
            or prefix["accepted_control_index"] != control
            or type(prefix.get("next_control_index")) is not int
            or prefix["next_control_index"] != control+1
            or any(prefix.get(k) is not False for k in
                   ("full_protocol_complete", "trajectory_parity_established", "physical_validation"))):
        raise ValueError("missing capture is not bounded by an incomplete accepted checkpoint prefix")
    checkpoint = regular_file(prefix.get("checkpoint_file", ""), 2*1024**3)
    metadata = regular_file(prefix.get("metadata_file", ""), 64*1024)
    if (digest(checkpoint) != prefix.get("checkpoint_sha256")
            or digest(metadata) != prefix.get("metadata_sha256")):
        raise ValueError("accepted checkpoint prefix reference hash mismatch")
    values = {}
    for line in metadata.read_text().splitlines():
        name, separator, value = line.partition("=")
        if not separator or name in values:
            raise ValueError("accepted checkpoint metadata has malformed/duplicate fields")
        values[name] = value
    if (values.get("schema") not in ("numi.cardboard.accepted-coupon-resume.v1",
                                     "numi.cardboard.accepted-coupon-resume.v2")
            or values.get("accepted_control_index") != str(control)
            or values.get("next_control_index") != str(control+1)
            or values.get("both_environments_accepted") != "true"
            or values.get("checkpoint_sha256") != prefix["checkpoint_sha256"]
            or values.get("checkpoint_bytes") != str(checkpoint.stat().st_size)):
        raise ValueError("accepted checkpoint metadata does not certify this prefix boundary")
    selectors = {
        "NM_FEM_OPERATOR_FD_CAPTURE_PATH": ("NM_FEM_OPERATOR_FD_SELECTOR", 2),
        "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH": ("NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", 6),
    }
    if key not in selectors:
        raise ValueError("this missing capture selector is not supported by coupon continuation")
    selector_key, fields = selectors[key]
    state = effective_contract["effective"][selector_key]
    raw = state.get("value") if state.get("set") else None
    parts = raw.split(":") if isinstance(raw, str) else []
    if (len(parts) != fields or any(not re.fullmatch(r"[0-9]+", p) or int(p)>UINT32_MAX for p in parts)
            or int(parts[0]) <= control):
        raise ValueError("missing diagnostic capture target is not strictly after the saved prefix")
    return {"key": key, "selector_key": selector_key, "selector": raw,
            "target_control": int(parts[0]), "accepted_control": control,
            "status": "not_yet_visited_at_accepted_checkpoint",
            "result_sha256": digest(result_path), "prefix_report_sha256": digest(prefix_path),
            "checkpoint_sha256": prefix["checkpoint_sha256"],
            "metadata_sha256": prefix["metadata_sha256"], "positive_qualification_allowed": False}

'''
name = 'cardboard_solver_environment.py'
needle = 'def validate_diagnostic_outputs(case_directory: os.PathLike[str] | str,\n                                capture_prefixes: Mapping[str, str]) -> dict[str, object]:'
assert texts[name].count(needle) == 1
texts[name] = texts[name].replace(needle, helper+'''def validate_diagnostic_outputs(case_directory: os.PathLike[str] | str,
                                capture_prefixes: Mapping[str, str], *,
                                native_argv: Sequence[str] = (), native_exit_code: int | None = None,
                                effective_contract: Mapping[str, object] | None = None) -> dict[str, object]:''')
needle = '    outputs: list[dict[str, object]] = []\n    total = 0\n'
assert texts[name].count(needle) == 1
texts[name] = texts[name].replace(needle, '    outputs: list[dict[str, object]] = []\n    pending: list[dict[str, object]] = []\n    total = 0\n')
needle = '        if not entries:\n            raise ValueError(f"diagnostic capture produced no files for {key}")\n'
assert texts[name].count(needle) == 1
texts[name] = texts[name].replace(needle, '''        if not entries:
            if not native_argv:
                raise ValueError(f"diagnostic capture produced no files for {key}")
            pending.append(_pending_checkpoint_capture(
                root, key, native_argv, native_exit_code, effective_contract))
            continue
''')
needle = '    return {"status": "validated", "files": outputs, "total_bytes": total,\n'
assert texts[name].count(needle) == 1
texts[name] = texts[name].replace(needle, '''    return {"status": "validated_with_pending_checkpoint_targets" if pending else "validated",
            "pending_captures": pending, "files": outputs, "total_bytes": total,
''')
name = 'cardboard_source_bound_run.py'
needle = '            case, diagnostic_prefixes)\n'
assert texts[name].count(needle) == 1
texts[name] = texts[name].replace(needle, '''            case, diagnostic_prefixes, native_argv=argv, native_exit_code=code,
            effective_contract=effective_environment_contract)
''')

# Tests use the checkout under test instead of the machine's retained overlay
# paths. An isolated candidate can explicitly supply a combined test source.
name = 'test_cardboard_solver_environment.py'
texts[name] = texts[name].replace('import unittest\n', 'import unittest\nimport os\n', 1)
start = texts[name].index('BASE = Path(__file__)')
end = texts[name].index('        contract = env.make_contract()', start)
texts[name] = texts[name][:start]+'''SOURCE_ROOT = Path(os.environ.get("CARDBOARD_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))


class EffectiveEnvironmentTests(unittest.TestCase):
    def test_registry_matches_checkout_and_detects_unregistered_control(self):
        env.validate_source_registry(SOURCE_ROOT)
        self.assertEqual(env.source_key_registry(SOURCE_ROOT), env.NATIVE_ENV_KEYS)
        self.assertEqual(len(env.NATIVE_ENV_KEYS), 49)
        self.assertIn("NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", env.NATIVE_ENV_KEYS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("matter/src/runtime.mm", "matter/tools/cardboard_probe.mm"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_ROOT / relative, target)
            with (root / "matter/src/runtime.mm").open("a") as stream:
                stream.write('\\nconst char *unregistered = "NM_FEM_UNREGISTERED_TEST_CONTROL";\\n')
            with self.assertRaisesRegex(ValueError, "unlisted=.*NM_FEM_UNREGISTERED_TEST_CONTROL"):
                env.validate_source_registry(root)
'''+texts[name][end:]
name = 'test_cardboard_resolution_study.py'
texts[name] = texts[name].replace('import unittest\n', 'import unittest\nimport os\n', 1)
start = texts[name].index('REPO_BASE = Path(__file__)')
end = texts[name].index('def combined_source_fixture', start)
texts[name] = texts[name][:start]+'''SOURCE_ROOT = Path(os.environ.get("CARDBOARD_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))


'''+texts[name][end:]
texts[name] = texts[name].replace('SOURCE160_PACKAGE / "source/matter/src/runtime.mm"', 'SOURCE_ROOT / "matter/src/runtime.mm"')
texts[name] = texts[name].replace('SOURCE157', 'SOURCE_ROOT')
texts[name] = texts[name].replace('Path("/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157")', 'SOURCE_ROOT')

patch = ''
for name, candidate in texts.items():
    relative = 'matter/tools/' + name
    path = O / 'candidate' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(candidate)
    original = (BASE / relative).read_text()
    patch += ''.join(difflib.unified_diff(original.splitlines(keepends=True), candidate.splitlines(keepends=True),
                    fromfile='a/'+relative, tofile='b/'+relative))
(O / 'overlay.patch').write_text(patch)
shutil.copy2(Path(__file__), O / 'prepare.py')
print(O)
