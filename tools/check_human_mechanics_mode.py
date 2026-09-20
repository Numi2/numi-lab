"""Structural presentation-boundary check; not a numerical qualification."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "apps/numilab_human_myosim_visual_probe.mm").read_text()
start = source.index("            if (!mechanicsOnly) {")
end = source.index("} // Rendering and visual qualification are absent in mechanics-only mode.", start)
render = source[start:end]
assert "metalrobo::MetalHybridRenderer renderer(rendererConfig);" in render
assert "require(completeVisualCoverage," in render
assert "requested NHTENDON2/3 attachment envelope is completely occluded" in render
assert "integratePersistentMetalHumanState(" not in render
assert source.index('myosim_articulated_mechanics=ok') > end
assert source.index('persistent_stand_trace={') > end
assert source.index('persistent_initial_force_reference=') > end
assert 'bool completeVisualCoverage = !mechanicsOnly;' in source
assert '"--mechanics-only requires --persistent-metal-stand"' in source
assert '"--mechanics-only cannot request a camera capture"' in source
assert '"myosim_articulated_marker_visual=ok"' in source
print("Mechanics-only presentation boundary retained; numerical evidence remains independent")
