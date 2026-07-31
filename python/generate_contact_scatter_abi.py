#!/usr/bin/env python3
"""Generate the shared host/Metal contact-scatter binding header."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(document: dict[str, object]) -> str:
    if document.get("schema") != 1:
        raise ValueError("contact-scatter schema version is invalid")
    version = document.get("abi_version")
    kernels = document.get("kernels")
    if not isinstance(version, int) or version <= 0:
        raise ValueError("contact-scatter ABI version is invalid")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("contact-scatter ABI has no kernels")
    declarations: list[str] = []
    assertions: list[str] = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("contact-scatter kernel is not an object")
        enum = kernel.get("enum")
        prefix = kernel.get("prefix")
        bindings = kernel.get("bindings")
        if (
            not isinstance(enum, str)
            or not enum.isidentifier()
            or not isinstance(prefix, str)
            or not prefix.startswith("MR_")
            or not isinstance(bindings, list)
            or not bindings
            or len(bindings) > 31
            or len(set(bindings)) != len(bindings)
            or any(
                not isinstance(binding, str)
                or not binding
                or not binding.replace("_", "").isalnum()
                or binding != binding.upper()
                for binding in bindings
            )
        ):
            raise ValueError(f"invalid contact-scatter kernel: {enum!r}")
        declarations.append(f"enum {enum} : mr_u32 {{")
        declarations.extend(
            f"    {prefix}{binding} = {index}u,"
            for index, binding in enumerate(bindings)
        )
        declarations.append(f"    {prefix}BUFFER_COUNT = {len(bindings)}u,")
        declarations.append("};")
        declarations.append("")
        assertions.append(
            f"static_assert({prefix}BUFFER_COUNT <= 31u);"
        )
    return "\n".join(
        [
            "// GENERATED FILE: python/generate_contact_scatter_abi.py",
            "#pragma once",
            "",
            '#include "metalrobo/gpu_types.h"',
            "",
            "// One schema owns host and Metal argument tables. Changing a",
            "// binding requires an ABI-version change.",
            f"#define MR_CONTACT_SCATTER_ABI_VERSION {version}u",
            "",
            *declarations,
            "#if defined(__cplusplus)",
            *assertions,
            "#endif",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render(json.loads(arguments.schema.read_text()))
    if arguments.check:
        if not arguments.output.is_file():
            raise SystemExit("generated contact-scatter ABI is missing")
        if arguments.output.read_text() != rendered:
            raise SystemExit("generated contact-scatter ABI is stale")
        return 0
    arguments.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
