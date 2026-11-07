"""Reuse the exact production shader recipe without duplicating it in a test."""
from pathlib import Path
import sys
root,output=map(Path,sys.argv[1:])
recipe=(root/"CMakeLists.txt").read_text()
start=recipe.index("set(METALROBO_SHADER_DIR")
end=recipe.index('add_custom_command(\n    OUTPUT "${METALROBO_NEURON_CULTURE_METALLIB}"',start)
Path(output).write_text(recipe[start:end].replace("${CMAKE_CURRENT_SOURCE_DIR}","${NUMI_WORLD_ROOT}"))
