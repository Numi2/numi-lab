#!/usr/bin/env python3
"""Quantization scale for the retained 128-node, 16 m rounded-input case."""
import math
import struct

def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]

EA = 2.3818718058345718
REST = .11758755785232791
RADIUS = .000020
DENSITY = 1100.0
H = f32(.001)
ULP = 2.0 ** -19
REFERENCE_FORCE = .098443843034694195
length = f32(REST / 127.0)
stiffness = f32(EA) / length
mass = 1.0 / f32(1.0 / (DENSITY * math.pi * RADIUS**2 * (REST / 127.0) / 2.0))
momentum_force = mass * (f32(.002) + H*f32(-1.2) - f32(.01)) / H
ideal_multiple = (length + (REFERENCE_FORCE - momentum_force) / stiffness) / ULP
nearest = round(ideal_multiple)
print(f"spacing_m={ULP:.17g} stiffness_N_per_m={stiffness:.17g} force_increment_N={stiffness*ULP:.17g}")
for multiple in (nearest-1, nearest, nearest+1):
    force = stiffness*(multiple*ULP-length)+momentum_force
    print(f"edge_spacing_units={multiple} force_N={force:.17g} error_N={abs(force-REFERENCE_FORCE):.17g}")
