#!/usr/bin/env python3
"""Check that the coupled static connector solution ignores borrowed pseudo mass."""

import math
import re
import subprocess
import sys


def run(executable: str, inverse_mass: float) -> dict[str, float]:
    output = subprocess.check_output(
        [executable, "--quasistatic", "--numerical-inverse-mass", str(inverse_mass)],
        text=True,
    )
    if "source_connector_coupled=accepted" not in output:
        raise AssertionError(output)
    values = {
        key: float(value)
        for key, value in re.findall(r"([a-z_]+)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", output)
    }
    for field in ("free_increment_x", "free_increment_y", "tied_node_x", "free_node_z"):
        if field not in values or not math.isfinite(values[field]):
            raise AssertionError(f"missing or invalid {field}: {output}")
    return values


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_source_quasistatic_mass.py EXECUTABLE")
    first = run(sys.argv[1], 1.0)
    second = run(sys.argv[1], 0.1)
    for field in ("free_increment_x", "free_increment_y", "tied_node_x", "free_node_z"):
        scale = max(abs(first[field]), abs(second[field]), 1.0e-6)
        if abs(first[field] - second[field]) > 2.0e-4 * scale:
            raise AssertionError(f"static {field} depends on pseudo inverse mass: "
                                 f"{first[field]} vs {second[field]}")
    print("source_quasistatic_pseudo_mass=independent "
          "source_knee_equivalence=unqualified")


if __name__ == "__main__":
    main()
