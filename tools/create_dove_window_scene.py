#!/usr/bin/env python3
"""Build cheap authored Numi-window geometry from the Deetjen surface archive."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    " ": ("00000",) * 7,
}


def write_surface_parts(surface: Path, output: Path) -> None:
    manifest = json.loads((surface / "manifest.json").read_text())
    topology = manifest["topology"]
    vertex_count = topology["vertexCount"]
    positions = struct.unpack(
        f"<{vertex_count * 3}f",
        (surface / manifest["binary"]["positions"]["file"]).read_bytes()[
            : vertex_count * 12
        ],
    )
    triangle_count = topology["triangleCount"]
    triangles = struct.unpack(
        f"<{triangle_count * 3}H",
        (surface / manifest["binary"]["triangles"]["file"]).read_bytes(),
    )
    transforms = {
        "body": lambda x, y, z: (x + 0.09, y, z + 0.035),
        "leftWing": lambda x, y, z: (x + 0.09, 3.3 * y - 0.24, z + 0.035),
        "rightWing": lambda x, y, z: (x + 0.09, 3.3 * y + 0.24, z + 0.035),
        "tail": lambda x, y, z: (x + 0.18, y, z + 0.04),
    }
    filenames = {
        "body": "body.obj",
        "leftWing": "left-wing.obj",
        "rightWing": "right-wing.obj",
        "tail": "tail.obj",
    }
    (output / "dove.mtl").write_text(
        "newmtl dove_white\nKd 0.78 0.80 0.84\nKs 0.10 0.10 0.10\nNs 20\n"
        "newmtl dove_wing\nKd 0.58 0.61 0.67\nKs 0.08 0.08 0.08\nNs 15\n"
        "newmtl dove_tail\nKd 0.45 0.48 0.54\nKs 0.06 0.06 0.06\nNs 10\n"
    )
    for component in topology["components"]:
        name = component["name"]
        vertex_offset = component["vertexOffset"]
        local_vertex_count = component["vertexCount"]
        triangle_offset = component["triangleOffset"]
        local_triangle_count = component["triangleCount"]
        transform = transforms[name]
        lines = ["mtllib dove.mtl", f"o deetjen_{name}"]
        material = "dove_white" if name == "body" else (
            "dove_tail" if name == "tail" else "dove_wing"
        )
        lines.append(f"usemtl {material}")
        for index in range(vertex_offset, vertex_offset + local_vertex_count):
            x, y, z = positions[index * 3 : index * 3 + 3]
            x, y, z = transform(x, y, z)
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")
        for index in range(triangle_offset, triangle_offset + local_triangle_count):
            a, b, c = triangles[index * 3 : index * 3 + 3]
            lines.append(
                f"f {a - vertex_offset + 1} {b - vertex_offset + 1} "
                f"{c - vertex_offset + 1}"
            )
        (output / filenames[name]).write_text("\n".join(lines) + "\n")


def usda_header() -> list[str]:
    # The measured Deetjen surface and authored flight hall are in meters and
    # MetalRobo is Z-up. Without explicit stage metadata Model I/O assumes
    # centimetres and Y-up, shrinking and rotating the complete presentation.
    return [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Scene"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
    ]


def write_room(output: Path) -> None:
    # Two complete phrases per row and eight rows per tile. Keeping phrase
    # boundaries inside the tile avoids the former "LAB NUMI" seam, while the
    # darker field preserves contrast against the light measured dove.
    width, height = 432, 448
    pixels = bytearray((56, 65, 82, 255) * (width * height))
    scale = 4
    phrase_width = len("NUMI LAB") * 6 * scale
    for origin_y in range(8, height, 56):
        for origin_x in range(8, width, phrase_width + 24):
            cursor = 0
            for letter in "NUMI LAB":
                for row, pattern in enumerate(FONT[letter]):
                    for column, filled in enumerate(pattern):
                        if filled != "1":
                            continue
                        for dy in range(scale):
                            for dx in range(scale):
                                x = origin_x + (cursor + column) * scale + dx
                                y = origin_y + row * scale + dy
                                if 0 <= x < width and 0 <= y < height:
                                    offset = (y * width + x) * 4
                                    pixels[offset : offset + 4] = bytes((24, 115, 255, 255))
                cursor += 6

    def png_chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 4 : (row + 1) * width * 4])
        for row in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(scanlines, 9))
    png += png_chunk(b"IEND", b"")
    (output / "numi-lab-surface.png").write_bytes(png)

    # Six quads only. Repeated UVs make NUMI LAB intrinsic to every wall,
    # floor, and ceiling surface rather than additional scene geometry.
    surfaces = {
        "floor": ((-8, -15, 0), (8, -15, 0), (8, 15, 0), (-8, 15, 0), 2, 4),
        "ceiling": ((-8, 15, 8), (8, 15, 8), (8, -15, 8), (-8, -15, 8), 2, 4),
        "south": ((-8, -15, 8), (8, -15, 8), (8, -15, 0), (-8, -15, 0), 2, 1),
        "north": ((8, 15, 8), (-8, 15, 8), (-8, 15, 0), (8, 15, 0), 2, 1),
        "west": ((-8, 15, 8), (-8, -15, 8), (-8, -15, 0), (-8, 15, 0), 4, 1),
        "east": ((8, -15, 8), (8, 15, 8), (8, 15, 0), (8, -15, 0), 4, 1),
    }
    lines = usda_header() + [
        'def Xform "Scene"', "{",
        '    def Material "numi_surface"', "    {",
        '        token outputs:surface.connect = </Scene/numi_surface/Surface.outputs:surface>',
        '        def Shader "UV"', "        {",
        '            uniform token info:id = "UsdPrimvarReader_float2"',
        '            token inputs:varname = "st"',
        '            float2 outputs:result', "        }",
        '        def Shader "Texture"', "        {",
        '            uniform token info:id = "UsdUVTexture"',
        '            asset inputs:file = @numi-lab-surface.png@',
        '            token inputs:sourceColorSpace = "sRGB"',
        '            token inputs:wrapS = "repeat"',
        '            token inputs:wrapT = "repeat"',
        '            float2 inputs:st.connect = </Scene/numi_surface/UV.outputs:result>',
        '            float3 outputs:rgb', "        }",
        '        def Shader "Surface"', "        {",
        '            uniform token info:id = "UsdPreviewSurface"',
        '            color3f inputs:diffuseColor.connect = </Scene/numi_surface/Texture.outputs:rgb>',
        '            color3f inputs:emissiveColor.connect = </Scene/numi_surface/Texture.outputs:rgb>',
        '            float inputs:opacity = 1.0',
        '            float inputs:roughness = 0.82',
        '            token outputs:surface', "        }", "    }",
    ]
    for name, (*points, repeat_u, repeat_v) in surfaces.items():
        lines.extend([
            f'    def Mesh "{name}"', "    {",
            '        uniform token subdivisionScheme = "none"',
            '        uniform bool doubleSided = true',
            '        rel material:binding = </Scene/numi_surface>',
            "        point3f[] points = [" +
            ", ".join(f"({x}, {y}, {z})" for x, y, z in points) + "]",
            "        int[] faceVertexCounts = [4]",
            "        int[] faceVertexIndices = [0, 1, 2, 3]",
            "        texCoord2f[] primvars:st = "
            f"[(0, 0), ({repeat_u}, 0), ({repeat_u}, {repeat_v}), (0, {repeat_v})] (",
            '            interpolation = "faceVarying"', "        )", "    }",
        ])
    lines.append("}")
    (output / "room.usda").write_text("\n".join(lines) + "\n")


def write_usda_from_obj(obj: Path, output: Path) -> None:
    vertices: list[tuple[float, float, float]] = []
    faces: dict[str, list[tuple[int, int, int]]] = {}
    material = "dove_white"
    for line in obj.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "usemtl":
            material = fields[1]
        elif fields[0] == "f":
            faces.setdefault(material, []).append(
                tuple(int(value.split("/")[0]) - 1 for value in fields[1:4])
            )
    colors = {
        "dove_white": (0.78, 0.80, 0.84),
        "dove_wing": (0.58, 0.61, 0.67),
        "dove_tail": (0.45, 0.48, 0.54),
        "floor": (0.08, 0.10, 0.14),
        "wall": (0.78, 0.80, 0.84),
        "numi_blue": (0.05, 0.32, 0.95),
    }
    lines = usda_header() + ['def Xform "Scene"', "{"]
    for name, material_faces in faces.items():
        used = sorted({index for face in material_faces for index in face})
        remap = {source: target for target, source in enumerate(used)}
        r, g, b = colors[name]
        safe = name.replace("-", "_")
        lines.extend([
            f'    def Material "{safe}_material"',
            "    {",
            f'        token outputs:surface.connect = </Scene/{safe}_material/Shader.outputs:surface>',
            '        def Shader "Shader"',
            "        {",
            '            uniform token info:id = "UsdPreviewSurface"',
            f"            color3f inputs:diffuseColor = ({r}, {g}, {b})",
            f"            color3f inputs:emissiveColor = ({0.55 * r}, {0.55 * g}, {0.55 * b})",
            "            float inputs:opacity = 1.0",
            "            float inputs:roughness = 0.72",
            "            token outputs:surface",
            "        }",
            "    }",
            f'    def Mesh "{safe}_mesh"',
            "    {",
            "        uniform token subdivisionScheme = \"none\"",
            "        uniform bool doubleSided = true",
            f"        rel material:binding = </Scene/{safe}_material>",
            "        point3f[] points = [",
        ])
        lines.extend(
            f"            ({vertices[index][0]}, {vertices[index][1]}, {vertices[index][2]}),"
            for index in used
        )
        lines.extend([
            "        ]",
            "        int[] faceVertexCounts = [" +
            ", ".join("3" for _ in material_faces) + "]",
            "        int[] faceVertexIndices = [" +
            ", ".join(
                str(remap[index])
                for face in material_faces
                for index in face
            ) + "]",
            "    }",
        ])
    lines.append("}")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_surface_parts(args.surface, args.output)
    write_room(args.output)
    for name in ("body", "left-wing", "right-wing", "tail"):
        write_usda_from_obj(
            args.output / f"{name}.obj",
            args.output / f"{name}.usda",
        )


if __name__ == "__main__":
    main()
