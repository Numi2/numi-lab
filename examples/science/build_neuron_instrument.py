#!/usr/bin/env python3
"""Build the existing native neuron owner sources in an isolated instrument directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    runtime, directory = args.runtime.resolve(), args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = [runtime / name for name in (
        "src/core/NeuronCulture.cpp", "src/core/NeuronCultureProtocol.cpp",
        "src/core/NeuronCultureEmbodiment.cpp", "src/core/NeuronCultureArtifacts.cpp",
        "src/metal/MetalNeuronCulture.mm", "apps/neuron_culture_probe.mm")]
    shader = runtime / "src/metal/NeuronCulture.metal"
    inputs = [*sources, shader, Path(__file__).resolve(), *sorted((runtime / "include/metalrobo").glob("*.h*"))]
    hashes = {str(path): sha(path) for path in inputs}
    commands = []
    def run(argv):
        commands.append(argv)
        subprocess.run(argv, check=True)
    air, metallib = directory / "NumiNeuron.air", directory / "NumiNeuron.metallib"
    run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math",
         "-Wall", "-Wextra", "-Werror", "-I", str(runtime / "include"), "-c", str(shader), "-o", str(air)])
    run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)])
    objects = []
    for source in sources:
        obj = directory / (source.stem + ".o")
        argv = ["xcrun", "clang++", "-std=c++20", "-O3", "-fno-fast-math", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                "-I", str(runtime / "include"), '-DMETALROBO_NEURON_METALLIB="' + str(metallib) + '"']
        if source.suffix == ".mm":
            argv.append("-fobjc-arc")
        run([*argv, "-c", str(source), "-o", str(obj)])
        objects.append(obj)
    binary = directory / "metalrobo_neuron_culture_probe"
    run(["xcrun", "clang++", *(str(obj) for obj in objects), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)])
    if hashes != {str(path): sha(path) for path in inputs}:
        raise RuntimeError("owner source changed during build")
    receipt = {"schema": "numi.science.native-instrument-build.v1", "runtime": str(runtime),
               "revision": subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip(),
               "compiler": subprocess.check_output(["xcrun", "clang++", "--version"], text=True).strip(),
               "sdk": subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True).strip(),
               "commands": commands, "inputs": hashes,
               "outputs": {str(binary): sha(binary), str(metallib): sha(metallib)},
               "scope": "Focused build of unmodified native owner sources; not the full Lab integration build"}
    (directory / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
