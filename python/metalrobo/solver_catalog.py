"""Numi Lab solver discovery and provenance-preserving profile configuration."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


DESCRIPTOR_SCHEMA = "numi.solver.v1"
CATALOG_SCHEMA = "numi.solver-catalog.v1"
FAMILY_CATALOG_SCHEMA = "numi.solver-family-catalog.v1"
FAMILY_SCHEMA = "numi.solver-family.v1"
PROFILE_SCHEMA = "numi.solver-profile.v1"
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PARAMETER_TYPES = {"boolean", "integer", "number", "string"}


class SolverConfigurationError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except OSError as error:
        raise SolverConfigurationError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SolverConfigurationError(
            f"invalid JSON in {path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error


def require_text(value: dict[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise SolverConfigurationError(f"{context}.{key} must be non-empty text")
    return result


def validate_parameter(name: str, spec: Any, context: str) -> None:
    if not IDENTIFIER.fullmatch(name):
        raise SolverConfigurationError(f"{context} has invalid parameter {name!r}")
    if not isinstance(spec, dict) or spec.get("type") not in PARAMETER_TYPES:
        raise SolverConfigurationError(
            f"{context}.{name} requires type boolean, integer, number, or string"
        )
    if "default" not in spec:
        raise SolverConfigurationError(f"{context}.{name} requires a default")
    for bound in ("minimum", "maximum"):
        if bound in spec and (
            not isinstance(spec[bound], (int, float))
            or isinstance(spec[bound], bool)
            or not math.isfinite(spec[bound])
        ):
            raise SolverConfigurationError(
                f"{context}.{name}.{bound} must be a finite number"
            )
    if (
        "minimum" in spec
        and "maximum" in spec
        and spec["minimum"] > spec["maximum"]
    ):
        raise SolverConfigurationError(
            f"{context}.{name}.minimum must not exceed maximum"
        )
    if "choices" in spec:
        choices = spec["choices"]
        if not isinstance(choices, list) or not choices:
            raise SolverConfigurationError(
                f"{context}.{name}.choices must be a non-empty list"
            )
        unconstrained = {key: value for key, value in spec.items() if key != "choices"}
        for choice in choices:
            validate_parameter_value(name, choice, unconstrained, context)
    validate_parameter_value(name, spec["default"], spec, context)


def validate_parameter_value(
    name: str, value: Any, spec: dict[str, Any], context: str
) -> None:
    kind = spec["type"]
    valid = (
        (kind == "boolean" and isinstance(value, bool))
        or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (
            kind == "number"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
        or (kind == "string" and isinstance(value, str))
    )
    if not valid:
        raise SolverConfigurationError(
            f"{context}.{name} must have type {kind}, got {value!r}"
        )
    if kind in {"integer", "number"} and not math.isfinite(value):
        raise SolverConfigurationError(
            f"{context}.{name} must be finite, got {value!r}"
        )
    if "choices" in spec and value not in spec["choices"]:
        raise SolverConfigurationError(
            f"{context}.{name} must be one of {spec['choices']}, got {value!r}"
        )
    if kind in {"integer", "number"}:
        if "minimum" in spec and value < spec["minimum"]:
            raise SolverConfigurationError(
                f"{context}.{name} must be >= {spec['minimum']}"
            )
        if "maximum" in spec and value > spec["maximum"]:
            raise SolverConfigurationError(
                f"{context}.{name} must be <= {spec['maximum']}"
            )


def validate_descriptor(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != DESCRIPTOR_SCHEMA:
        raise SolverConfigurationError(f"{context} must use schema {DESCRIPTOR_SCHEMA}")
    solver_id = require_text(value, "id", context)
    if not IDENTIFIER.fullmatch(solver_id):
        raise SolverConfigurationError(f"{context}.id is invalid: {solver_id!r}")
    family = value.get("family")
    if not isinstance(family, dict):
        raise SolverConfigurationError(f"{context}.family must be an object")
    family_id = require_text(family, "id", f"{context}.family")
    variant = require_text(family, "variant", f"{context}.family")
    if not IDENTIFIER.fullmatch(family_id) or not IDENTIFIER.fullmatch(variant):
        raise SolverConfigurationError(
            f"{context}.family contains an invalid identifier"
        )
    require_text(family, "name", f"{context}.family")
    require_text(family, "summary", f"{context}.family")
    if not isinstance(family.get("default"), bool):
        raise SolverConfigurationError(
            f"{context}.family.default must be boolean"
        )
    for key in (
        "name",
        "domain",
        "summary",
        "backend",
        "precision",
        "role",
        "availability",
    ):
        require_text(value, key, context)
    owner = value.get("owner")
    if not isinstance(owner, dict):
        raise SolverConfigurationError(f"{context}.owner must be an object")
    for key in ("implementation", "configuration", "documentation"):
        require_text(owner, key, f"{context}.owner")
    selection = value.get("selection")
    if not isinstance(selection, dict):
        raise SolverConfigurationError(f"{context}.selection must be an object")
    require_text(selection, "kind", f"{context}.selection")
    require_text(selection, "selector", f"{context}.selection")
    targets = selection.get("targets")
    if not isinstance(targets, list) or not targets or not all(
        isinstance(target, str) and target for target in targets
    ):
        raise SolverConfigurationError(
            f"{context}.selection.targets must be a non-empty text list"
        )
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise SolverConfigurationError(f"{context}.parameters must be an object")
    for name, spec in parameters.items():
        validate_parameter(name, spec, f"{context}.parameters")
    evidence = value.get("evidence_boundary")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item for item in evidence
    ):
        raise SolverConfigurationError(
            f"{context}.evidence_boundary must be a non-empty text list"
        )
    return value


def validate_profile(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != PROFILE_SCHEMA:
        raise SolverConfigurationError(f"{context} must use schema {PROFILE_SCHEMA}")
    profile_id = require_text(value, "id", context)
    solver_id = require_text(value, "solver_id", context)
    if not IDENTIFIER.fullmatch(profile_id) or not IDENTIFIER.fullmatch(solver_id):
        raise SolverConfigurationError(f"{context} contains an invalid identifier")
    for key in ("family_id", "variant"):
        if key in value:
            identifier = require_text(value, key, context)
            if not IDENTIFIER.fullmatch(identifier):
                raise SolverConfigurationError(
                    f"{context}.{key} is an invalid identifier"
                )
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise SolverConfigurationError(f"{context}.provenance must be an object")
    descriptor_hash = require_text(
        provenance, "descriptor_sha256", f"{context}.provenance"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", descriptor_hash):
        raise SolverConfigurationError(
            f"{context}.provenance.descriptor_sha256 is invalid"
        )
    require_text(provenance, "descriptor_source", f"{context}.provenance")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise SolverConfigurationError(f"{context}.parameters must be an object")
    return value


def group_solver_families(
    descriptors: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        metadata = descriptor["family"]
        family_id = metadata["id"]
        if family_id not in families:
            families[family_id] = {
                "schema": FAMILY_SCHEMA,
                "id": family_id,
                "name": metadata["name"],
                "summary": metadata["summary"],
                "implementations": [],
            }
        family = families[family_id]
        if (
            family["name"] != metadata["name"]
            or family["summary"] != metadata["summary"]
        ):
            raise SolverConfigurationError(
                f"solver family {family_id!r} has inconsistent metadata"
            )
        family["implementations"].append(descriptor)

    for family_id, family in families.items():
        implementations = family["implementations"]
        variants = [item["family"]["variant"] for item in implementations]
        if len(variants) != len(set(variants)):
            raise SolverConfigurationError(
                f"solver family {family_id!r} has duplicate variant IDs"
            )
        defaults = [item for item in implementations if item["family"]["default"]]
        if len(defaults) != 1:
            raise SolverConfigurationError(
                f"solver family {family_id!r} requires exactly one default variant"
            )
        implementations.sort(key=lambda item: item["family"]["variant"])
        family["default_implementation"] = defaults[0]["id"]
        family["default_variant"] = defaults[0]["family"]["variant"]
    return families


class SolverCatalog:
    def __init__(self, arguments: argparse.Namespace):
        self.runtime_root = arguments.runtime_root.resolve()
        self.workspace = arguments.workspace.resolve()
        self.catalog_path = arguments.catalog.resolve()
        config_home = Path(
            os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        ).expanduser()
        self.user_root = config_home / "numi"
        self.descriptor_paths = [self.workspace / ".numi" / "solvers"]
        self.descriptor_paths.extend(
            Path(path).expanduser()
            for path in os.environ.get("NUMI_SOLVER_PATH", "").split(":")
            if path
        )
        self.descriptor_paths.append(self.user_root / "solvers")
        self.profile_paths = [
            self.workspace / ".numi" / "profiles" / "solvers",
            self.user_root / "profiles" / "solvers",
        ]

    def _descriptors_from_file(self, path: Path) -> list[dict[str, Any]]:
        value = read_json(path)
        if isinstance(value, dict) and value.get("schema") == CATALOG_SCHEMA:
            solvers = value.get("solvers")
            if not isinstance(solvers, list):
                raise SolverConfigurationError(f"{path}.solvers must be a list")
            result = [
                validate_descriptor(solver, f"{path}:solvers[{index}]")
                for index, solver in enumerate(solvers)
            ]
        else:
            result = [validate_descriptor(value, str(path))]
        identifiers = [solver["id"] for solver in result]
        if len(identifiers) != len(set(identifiers)):
            raise SolverConfigurationError(f"{path} contains duplicate solver IDs")
        return result

    def descriptors(self) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        sources: list[Path] = []
        for directory in self.descriptor_paths:
            if directory.is_dir():
                sources.extend(sorted(directory.glob("*.json")))
        sources.append(self.catalog_path)
        for path in sources:
            for descriptor in self._descriptors_from_file(path):
                if descriptor["id"] in resolved:
                    continue
                enriched = dict(descriptor)
                enriched["_provenance"] = {
                    "source": str(path.resolve()),
                    "sha256": fingerprint(descriptor),
                }
                resolved[descriptor["id"]] = enriched
        return resolved

    def descriptor(self, solver_id: str) -> dict[str, Any]:
        try:
            return self.descriptors()[solver_id]
        except KeyError as error:
            raise SolverConfigurationError(
                f"unknown solver {solver_id!r}; run `numi solvers list`"
            ) from error

    def families(self) -> dict[str, dict[str, Any]]:
        return group_solver_families(list(self.descriptors().values()))

    def resolve(
        self,
        reference: str,
        *,
        variant: str | None = None,
        target: str | None = None,
    ) -> dict[str, Any]:
        descriptors = self.descriptors()
        families = group_solver_families(list(descriptors.values()))
        if reference in descriptors:
            if variant is not None:
                raise SolverConfigurationError(
                    "--variant is only valid with a solver family ID"
                )
            candidates = [descriptors[reference]]
        else:
            family = families.get(reference)
            if family is None:
                choices = sorted([*families, *descriptors])
                close = difflib.get_close_matches(reference, choices, n=1)
                suggestion = f"; did you mean {close[0]!r}?" if close else ""
                raise SolverConfigurationError(
                    f"unknown solver family or implementation {reference!r}; "
                    f"run `numi solvers list`{suggestion}"
                )
            candidates = list(family["implementations"])
            if variant is not None:
                candidates = [
                    item
                    for item in candidates
                    if item["family"]["variant"] == variant or item["id"] == variant
                ]

        if target is not None:
            candidates = [
                item
                for item in candidates
                if any(
                    target.casefold() in candidate_target.casefold()
                    for candidate_target in item["selection"]["targets"]
                )
            ]
        if not candidates:
            detail = f" variant {variant!r}" if variant else ""
            target_detail = f" for target {target!r}" if target else ""
            raise SolverConfigurationError(
                f"{reference!r} has no{detail} implementation{target_detail}"
            )
        if len(candidates) == 1:
            return candidates[0]
        defaults = [item for item in candidates if item["family"]["default"]]
        if len(defaults) == 1:
            return defaults[0]
        choices = ", ".join(
            f"{item['family']['variant']} ({item['id']})" for item in candidates
        )
        raise SolverConfigurationError(
            f"{reference!r} is ambiguous; use --variant with one of: {choices}"
        )

    def profile_path(self, reference: str) -> Path:
        explicit = Path(reference).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        for directory in self.profile_paths:
            candidate = directory / f"{reference}.json"
            if candidate.is_file():
                return candidate.resolve()
        raise SolverConfigurationError(
            f"unknown solver profile {reference!r}; configure or pass a path"
        )


def without_internal(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise SolverConfigurationError(f"--set requires KEY=VALUE, got {text!r}")
    key, raw = text.split("=", 1)
    if not IDENTIFIER.fullmatch(key):
        raise SolverConfigurationError(f"invalid parameter name {key!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def write_new_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise SolverConfigurationError(f"refusing to overwrite existing {path}")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SolverConfigurationError(f"cannot write {path}: {error}") from error


def matches_filters(descriptor: dict[str, Any], arguments: argparse.Namespace) -> bool:
    return (
        arguments.domain is None or descriptor["domain"] == arguments.domain
    ) and (
        arguments.target is None
        or any(
            arguments.target.casefold() in target.casefold()
            for target in descriptor["selection"]["targets"]
        )
    )


def family_view(
    family: dict[str, Any], implementations: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    selected = implementations or family["implementations"]
    variants = [
        {
            "id": item["family"]["variant"],
            "implementation_id": item["id"],
            "backend": item["backend"],
            "precision": item["precision"],
            "role": item["role"],
            "availability": item["availability"],
            "domain": item["domain"],
            "targets": item["selection"]["targets"],
            "default": item["family"]["default"],
        }
        for item in selected
    ]
    return {
        "schema": FAMILY_SCHEMA,
        "id": family["id"],
        "name": family["name"],
        "summary": family["summary"],
        "default_variant": family["default_variant"],
        "default_implementation": family["default_implementation"],
        "variants": variants,
    }


def command_list(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    all_descriptors = catalog.descriptors()
    families = group_solver_families(list(all_descriptors.values()))
    descriptors = [
        descriptor
        for descriptor in all_descriptors.values()
        if matches_filters(descriptor, arguments)
    ]
    descriptors.sort(key=lambda item: (item["domain"], item["id"]))
    if arguments.implementations:
        if arguments.json:
            print(
                json.dumps(
                    {"schema": CATALOG_SCHEMA, "solvers": descriptors},
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if not descriptors:
            raise SolverConfigurationError("no implementations match the filters")
        width = max(len(item["id"]) for item in descriptors)
        for item in descriptors:
            print(
                f"{item['id']:<{width}}  {item['backend']:<8}  "
                f"{item['availability']:<12}  {item['summary']}"
            )
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for descriptor in descriptors:
        grouped.setdefault(descriptor["family"]["id"], []).append(descriptor)
    views = [
        family_view(families[family_id], implementations)
        for family_id, implementations in sorted(grouped.items())
    ]
    if arguments.json:
        print(
            json.dumps(
                {"schema": FAMILY_CATALOG_SCHEMA, "families": views},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not views:
        raise SolverConfigurationError("no solver families match the filters")
    width = max(len(item["id"]) for item in views)
    for item in views:
        variant_count = len(item["variants"])
        variant_label = "variant" if variant_count == 1 else "variants"
        print(
            f"{item['id']:<{width}}  {variant_count} {variant_label:<8}  "
            f"{item['summary']}"
        )


def command_inspect(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    families = catalog.families()
    if (
        arguments.solver in families
        and arguments.variant is None
        and arguments.target is None
    ):
        family = families[arguments.solver]
        view = family_view(family)
        if arguments.json:
            print(json.dumps(view, indent=2, sort_keys=True))
            return
        print(f"{family['name']} ({family['id']})")
        print(f"Summary: {family['summary']}")
        print(
            f"Default: {family['default_variant']} "
            f"({family['default_implementation']})"
        )
        print("Variants:")
        for item in family["implementations"]:
            marker = "*" if item["family"]["default"] else " "
            print(
                f" {marker} {item['family']['variant']:<30} "
                f"{item['backend']}/{item['precision']}  {item['id']}"
            )
            print(f"     targets: {', '.join(item['selection']['targets'])}")
        print("Inspect an implementation ID for parameters and evidence boundaries.")
        return

    descriptor = catalog.resolve(
        arguments.solver,
        variant=arguments.variant,
        target=arguments.target,
    )
    if arguments.json:
        print(json.dumps(descriptor, indent=2, sort_keys=True))
        return
    print(f"{descriptor['name']} ({descriptor['id']})")
    print(f"Domain:       {descriptor['domain']}")
    print(
        f"Family:       {descriptor['family']['id']} / "
        f"{descriptor['family']['variant']}"
    )
    print(f"Backend:      {descriptor['backend']} / {descriptor['precision']}")
    print(f"Role:         {descriptor['role']}")
    print(f"Availability: {descriptor['availability']}")
    print(f"Summary:      {descriptor['summary']}")
    print(f"Configuration: {descriptor['owner']['configuration']}")
    print(f"Implementation: {descriptor['owner']['implementation']}")
    print(f"Documentation: {descriptor['owner']['documentation']}")
    print(f"Selection:    {descriptor['selection']['kind']}")
    print(f"Selector:     {descriptor['selection']['selector']}")
    print("Targets:")
    for target in descriptor["selection"]["targets"]:
        print(f"  {target}")
    print("Parameters:")
    if descriptor["parameters"]:
        for name, spec in descriptor["parameters"].items():
            print(f"  {name}={json.dumps(spec['default'])} ({spec['type']})")
    else:
        print("  none")
    print("Evidence boundary:")
    for boundary in descriptor["evidence_boundary"]:
        print(f"  {boundary}")
    print(f"Source:       {descriptor['_provenance']['source']}")
    print(f"SHA-256:      {descriptor['_provenance']['sha256']}")


def command_configure(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    descriptor = catalog.resolve(
        arguments.solver,
        variant=arguments.variant,
        target=arguments.target,
    )
    parameter_specs = descriptor["parameters"]
    parameters = {
        name: spec["default"] for name, spec in parameter_specs.items()
    }
    for text in arguments.set_values:
        name, value = parse_override(text)
        if name not in parameter_specs:
            raise SolverConfigurationError(
                f"{descriptor['id']} has no parameter {name!r}"
            )
        validate_parameter_value(
            name, value, parameter_specs[name], f"profile {arguments.profile}"
        )
        parameters[name] = value
    profile = {
        "schema": PROFILE_SCHEMA,
        "id": arguments.profile,
        "family_id": descriptor["family"]["id"],
        "variant": descriptor["family"]["variant"],
        "solver_id": descriptor["id"],
        "parameters": parameters,
        "selection": descriptor["selection"],
        "provenance": {
            "descriptor_source": descriptor["_provenance"]["source"],
            "descriptor_sha256": descriptor["_provenance"]["sha256"],
        },
    }
    validate_profile(profile, f"profile {arguments.profile}")
    if arguments.scope == "workspace":
        root = catalog.profile_paths[0]
    else:
        root = catalog.profile_paths[1]
    path = root / f"{arguments.profile}.json"
    write_new_json(path, profile)
    print(
        f"Configured {descriptor['family']['id']} / "
        f"{descriptor['family']['variant']} as {arguments.profile}"
    )
    print(f"Implementation: {descriptor['id']}")
    print(f"Profile: {path}")
    print(f"Descriptor SHA-256: {descriptor['_provenance']['sha256']}")


def command_show(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    path = catalog.profile_path(arguments.profile)
    profile = validate_profile(read_json(path), str(path))
    descriptor = catalog.descriptor(profile["solver_id"])
    if profile.get("family_id", descriptor["family"]["id"]) != (
        descriptor["family"]["id"]
    ) or profile.get("variant", descriptor["family"]["variant"]) != (
        descriptor["family"]["variant"]
    ):
        raise SolverConfigurationError(
            f"{path} family or variant differs from the current descriptor"
        )
    expected_parameters = set(descriptor["parameters"])
    provided_parameters = set(profile["parameters"])
    if provided_parameters != expected_parameters:
        missing = sorted(expected_parameters - provided_parameters)
        extra = sorted(provided_parameters - expected_parameters)
        raise SolverConfigurationError(
            f"{path} parameter set differs from the descriptor; "
            f"missing={missing}, extra={extra}"
        )
    if profile.get("selection") != descriptor["selection"]:
        raise SolverConfigurationError(
            f"{path} selection targets differ from the current descriptor"
        )
    for name, value in profile["parameters"].items():
        if name not in descriptor["parameters"]:
            raise SolverConfigurationError(
                f"{path} parameter {name!r} is absent from the current descriptor"
            )
        validate_parameter_value(
            name, value, descriptor["parameters"][name], str(path)
        )
    current_hash = descriptor["_provenance"]["sha256"]
    recorded_hash = profile["provenance"]["descriptor_sha256"]
    status = "current" if current_hash == recorded_hash else "stale"
    output = dict(profile)
    output["status"] = status
    output["current_descriptor_sha256"] = current_hash
    output["profile_path"] = str(path)
    if arguments.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Profile: {profile['id']}")
        print(
            f"Solver:  {descriptor['family']['id']} / "
            f"{descriptor['family']['variant']}"
        )
        print(f"Implementation: {profile['solver_id']}")
        print(f"Status:  {status}")
        print(f"Path:    {path}")
        for name, value in profile["parameters"].items():
            print(f"  {name}={json.dumps(value)}")
    if status != "current":
        raise SolverConfigurationError(
            "profile descriptor has drifted; inspect and create a new profile"
        )


def command_validate(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    path = Path(arguments.path).expanduser().resolve()
    value = read_json(path)
    if isinstance(value, dict) and value.get("schema") == PROFILE_SCHEMA:
        profile = validate_profile(value, str(path))
        descriptor = catalog.descriptor(profile["solver_id"])
        if profile.get("family_id", descriptor["family"]["id"]) != (
            descriptor["family"]["id"]
        ) or profile.get("variant", descriptor["family"]["variant"]) != (
            descriptor["family"]["variant"]
        ):
            raise SolverConfigurationError(
                f"{path} family or variant differs from the current descriptor"
            )
        if set(profile["parameters"]) != set(descriptor["parameters"]):
            raise SolverConfigurationError(
                f"{path} parameter set differs from the current descriptor"
            )
        for name, parameter_value in profile["parameters"].items():
            validate_parameter_value(
                name, parameter_value, descriptor["parameters"][name], str(path)
            )
        if profile.get("selection") != descriptor["selection"]:
            raise SolverConfigurationError(
                f"{path} selection targets differ from the current descriptor"
            )
        if profile["provenance"]["descriptor_sha256"] != (
            descriptor["_provenance"]["sha256"]
        ):
            raise SolverConfigurationError(
                f"{path} descriptor fingerprint is stale"
            )
        kind = "profile"
    elif isinstance(value, dict) and value.get("schema") == CATALOG_SCHEMA:
        descriptors = catalog._descriptors_from_file(path)
        group_solver_families(descriptors)
        kind = "catalog"
    else:
        validate_descriptor(value, str(path))
        kind = "descriptor"
    print(f"Valid {kind}: {path}")


def command_register(catalog: SolverCatalog, arguments: argparse.Namespace) -> None:
    source = Path(arguments.path).expanduser().resolve()
    descriptor = validate_descriptor(read_json(source), str(source))
    if arguments.scope == "workspace":
        root = catalog.descriptor_paths[0]
    else:
        root = catalog.user_root / "solvers"
    destination = root / f"{descriptor['id']}.json"
    prospective = catalog.descriptors()
    prospective[descriptor["id"]] = descriptor
    group_solver_families(list(prospective.values()))
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise SolverConfigurationError(
                f"refusing to overwrite existing {destination}"
            )
        shutil.copyfile(source, destination)
    except OSError as error:
        raise SolverConfigurationError(
            f"cannot register {source} as {destination}: {error}"
        ) from error
    print(f"Registered {descriptor['id']}")
    print(f"Descriptor: {destination}")


def command_paths(catalog: SolverCatalog, _: argparse.Namespace) -> None:
    print("Solver descriptors (first ID wins):")
    for path in catalog.descriptor_paths:
        print(f"  {path}")
    print(f"  {catalog.catalog_path} (bundled)")
    print("Solver profiles:")
    for path in catalog.profile_paths:
        print(f"  {path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(add_help=False)
    root.add_argument("--runtime-root", required=True, type=Path)
    root.add_argument("--workspace", required=True, type=Path)
    root.add_argument("--catalog", required=True, type=Path)
    commands = root.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--domain")
    list_parser.add_argument("--target")
    list_parser.add_argument("--implementations", action="store_true")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(handler=command_list)

    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("solver")
    inspect_parser.add_argument("--variant")
    inspect_parser.add_argument("--target")
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(handler=command_inspect)

    configure_parser = commands.add_parser("configure")
    configure_parser.add_argument("solver")
    configure_parser.add_argument("--variant")
    configure_parser.add_argument("--target")
    configure_parser.add_argument("--profile", required=True)
    configure_parser.add_argument(
        "--scope", choices=("workspace", "user"), default="workspace"
    )
    configure_parser.add_argument("--set", dest="set_values", action="append", default=[])
    configure_parser.set_defaults(handler=command_configure)

    show_parser = commands.add_parser("show")
    show_parser.add_argument("profile")
    show_parser.add_argument("--json", action="store_true")
    show_parser.set_defaults(handler=command_show)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(handler=command_validate)

    register_parser = commands.add_parser("register")
    register_parser.add_argument("path")
    register_parser.add_argument(
        "--scope", choices=("workspace", "user"), default="workspace"
    )
    register_parser.set_defaults(handler=command_register)

    paths_parser = commands.add_parser("paths")
    paths_parser.set_defaults(handler=command_paths)
    return root


def main() -> int:
    arguments = parser().parse_args()
    try:
        arguments.handler(SolverCatalog(arguments), arguments)
        return 0
    except SolverConfigurationError as error:
        print(f"numi solvers: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
