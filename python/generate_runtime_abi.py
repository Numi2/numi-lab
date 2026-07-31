#!/usr/bin/env python3
"""Generate the native runtime ABI shared by C++, Metal, and Swift."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LIFETIMES = ("boundary", "immutable", "persistent", "transient")


def _words(symbol: str) -> str:
    name = symbol[1:] if symbol.startswith("k") else symbol
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return name.lower()


def _validate(document: dict[str, object]) -> tuple[
    int,
    int,
    list[tuple[str, int]],
    dict[str, str],
    set[str],
    list[dict[str, object]],
]:
    if document.get("schema") != 1:
        raise ValueError("runtime ABI schema version is invalid")
    version = document.get("abi_version")
    world = document.get("world_buffers")
    kernels = document.get("kernels")
    if not isinstance(version, int) or version <= 0:
        raise ValueError("runtime ABI version is invalid")
    if not isinstance(world, dict) or not isinstance(kernels, list):
        raise ValueError("runtime ABI sections are missing")

    count = world.get("count")
    entries = world.get("entries")
    classifications = world.get("classifications")
    persistent_inputs = world.get("persistent_inputs")
    if (
        not isinstance(count, int)
        or count <= 0
        or not isinstance(entries, list)
        or not isinstance(classifications, dict)
        or not isinstance(persistent_inputs, list)
    ):
        raise ValueError("world-buffer schema is invalid")

    parsed: list[tuple[str, int]] = []
    symbols: set[str] = set()
    indices: set[int] = set()
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or not entry[0].startswith("k")
            or not entry[0].isidentifier()
            or not isinstance(entry[1], int)
            or entry[1] < 0
            or entry[1] >= count
            or entry[0] in symbols
            or entry[1] in indices
        ):
            raise ValueError(f"invalid world-buffer entry: {entry!r}")
        parsed.append((entry[0], entry[1]))
        symbols.add(entry[0])
        indices.add(entry[1])
    parsed.sort(key=lambda item: item[1])

    lifetime_by_symbol = {symbol: "boundary" for symbol in symbols}
    assigned: set[str] = set()
    for lifetime, names in classifications.items():
        if lifetime not in LIFETIMES[1:] or not isinstance(names, list):
            raise ValueError(f"invalid world-buffer lifetime: {lifetime!r}")
        for symbol in names:
            if (
                not isinstance(symbol, str)
                or symbol not in symbols
                or symbol in assigned
            ):
                raise ValueError(
                    f"invalid or duplicate lifetime assignment: {symbol!r}"
                )
            assigned.add(symbol)
            lifetime_by_symbol[symbol] = lifetime

    persistent_input_set = set(persistent_inputs)
    if len(persistent_input_set) != len(persistent_inputs):
        raise ValueError("persistent-input list contains duplicates")
    for symbol in persistent_input_set:
        if lifetime_by_symbol.get(symbol) != "persistent":
            raise ValueError(
                f"persistent input is not persistent: {symbol!r}"
            )

    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("runtime kernel is not an object")
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
                or binding != binding.upper()
                or not binding.replace("_", "").isalnum()
                for binding in bindings
            )
        ):
            raise ValueError(f"invalid runtime kernel: {enum!r}")

    return (
        version,
        count,
        parsed,
        lifetime_by_symbol,
        persistent_input_set,
        kernels,
    )


def render_header(document: dict[str, object]) -> str:
    (
        version,
        count,
        entries,
        lifetime_by_symbol,
        persistent_inputs,
        kernels,
    ) = _validate(document)
    kernel_declarations: list[str] = []
    kernel_assertions: list[str] = []
    for kernel in kernels:
        enum = kernel["enum"]
        prefix = kernel["prefix"]
        bindings = kernel["bindings"]
        kernel_declarations.append(f"enum {enum} : mr_u32 {{")
        kernel_declarations.extend(
            f"    {prefix}{binding} = {index}u,"
            for index, binding in enumerate(bindings)
        )
        kernel_declarations.append(
            f"    {prefix}BUFFER_COUNT = {len(bindings)}u,"
        )
        kernel_declarations.append("};")
        kernel_declarations.append("")
        kernel_assertions.append(
            f"static_assert({prefix}BUFFER_COUNT <= 31u);"
        )

    symbol_by_index = {index: symbol for symbol, index in entries}
    lifetime_rows = []
    persistent_rows = []
    debug_rows = []
    for index in range(count):
        symbol = symbol_by_index.get(index)
        if symbol is None:
            lifetime_rows.append("        BufferLifetime::unused,")
            persistent_rows.append("        false,")
            debug_rows.append(f'        "reserved {index}",')
            continue
        lifetime_rows.append(
            f"        BufferLifetime::{lifetime_by_symbol[symbol]},"
        )
        persistent_rows.append(
            "        true," if symbol in persistent_inputs else "        false,"
        )
        debug_rows.append(f'        "{_words(symbol)}",')

    return "\n".join(
        [
            "// GENERATED FILE: python/generate_runtime_abi.py",
            "#pragma once",
            "",
            '#include "metalrobo/gpu_types.h"',
            "",
            "// One schema owns the native resource table and shared kernel",
            "// bindings. Any persisted layout change increments this version.",
            f"#define MR_RUNTIME_ABI_VERSION {version}u",
            "",
            *kernel_declarations,
            "#if defined(__cplusplus) && !defined(__METAL_VERSION__)",
            "#include <array>",
            "#include <cstddef>",
            "#include <cstdint>",
            "",
            "namespace metalrobo::runtime_abi {",
            "",
            "enum BufferIndex : std::size_t {",
            *[
                f"    {symbol} = {index}u,"
                for symbol, index in entries
            ],
            f"    kRawBufferCount = {count}u,",
            "};",
            "",
            "enum class BufferLifetime : std::uint8_t {",
            "    boundary = 0u,",
            "    immutable = 1u,",
            "    persistent = 2u,",
            "    transient = 3u,",
            "    unused = 4u,",
            "};",
            "",
            "inline constexpr std::array<BufferLifetime, kRawBufferCount>",
            "    kBufferLifetimes{{",
            *lifetime_rows,
            "    }};",
            "inline constexpr std::array<bool, kRawBufferCount>",
            "    kPersistentInputs{{",
            *persistent_rows,
            "    }};",
            "inline constexpr std::array<const char*, kRawBufferCount>",
            "    kBufferDebugNames{{",
            *debug_rows,
            "    }};",
            "",
            "[[nodiscard]] constexpr bool validBufferIndex(",
            "    const std::size_t index",
            ") noexcept {",
            "    return index < kRawBufferCount &&",
            "        kBufferLifetimes[index] != BufferLifetime::unused;",
            "}",
            "",
            "[[nodiscard]] constexpr bool isImmutable(",
            "    const std::size_t index",
            ") noexcept {",
            "    return index < kRawBufferCount &&",
            "        kBufferLifetimes[index] == BufferLifetime::immutable;",
            "}",
            "",
            "[[nodiscard]] constexpr bool isPersistent(",
            "    const std::size_t index",
            ") noexcept {",
            "    return index < kRawBufferCount &&",
            "        kBufferLifetimes[index] == BufferLifetime::persistent;",
            "}",
            "",
            "[[nodiscard]] constexpr bool isPersistentInput(",
            "    const std::size_t index",
            ") noexcept {",
            "    return index < kRawBufferCount && kPersistentInputs[index];",
            "}",
            "",
            "[[nodiscard]] constexpr bool isTransient(",
            "    const std::size_t index",
            ") noexcept {",
            "    return index < kRawBufferCount &&",
            "        kBufferLifetimes[index] == BufferLifetime::transient;",
            "}",
            "",
            "static_assert(kBufferLifetimes.size() == kRawBufferCount);",
            "static_assert(kPersistentInputs.size() == kRawBufferCount);",
            "static_assert(kBufferDebugNames.size() == kRawBufferCount);",
            *kernel_assertions,
            "",
            "} // namespace metalrobo::runtime_abi",
            "#endif",
            "",
        ]
    )


def render_swift(document: dict[str, object]) -> str:
    version, count, entries, _, _, _ = _validate(document)
    symbol_by_index = {index: symbol for symbol, index in entries}
    names = [
        _words(symbol_by_index[index])
        if index in symbol_by_index
        else f"reserved {index}"
        for index in range(count)
    ]
    return "\n".join(
        [
            "// GENERATED FILE: python/generate_runtime_abi.py",
            "import Foundation",
            "",
            "enum MetalRoboRuntimeABI {",
            f"    static let version: UInt32 = {version}",
            f"    static let worldBufferCount = {count}",
            "    static let worldBufferDebugNames: [String] = [",
            *[f'        "{name}",' for name in names],
            "    ]",
            "}",
            "",
        ]
    )


def _check(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"generated {label} is missing")
    if path.read_text() != expected:
        raise SystemExit(f"generated {label} is stale")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--swift", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    document = json.loads(arguments.schema.read_text())
    header = render_header(document)
    swift = render_swift(document)
    if arguments.check:
        _check(arguments.header, header, "runtime ABI header")
        _check(arguments.swift, swift, "runtime ABI Swift source")
        return 0
    arguments.header.write_text(header)
    arguments.swift.write_text(swift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
