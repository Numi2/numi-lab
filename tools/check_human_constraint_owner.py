"""Read-only structural regression for the corrected Human force owners.

Numerical checks are the shared C++ executables and native Metal compilation.
This guard does not certify physical standing or temporal convergence.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
kernel = (root / "src/metal/NumiHumanStand.metal").read_text()
runner = (root / "apps/numilab_human_myosim_visual_probe.mm").read_text()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


for symbol in ("mrNumiHumanSupportSeedImpulse(", "mrNumiHumanContactVelocityTarget(",
               "mrNumiHumanProjectIntervalImpulse(", "mrNumiHumanProjectEqualityLimitBlock("):
    require(symbol in kernel, f"production standing owner lost {symbol}")
require("effort += supportForce" not in kernel,
        "support reaction reintroduced as a permanent effort")
require("lambdas[3u * contact + 0u] = seed;" in kernel,
        "normal warm-start impulse is not registered")
require("limitAccumulatedImpulses[limit] = nextImpulse;" in kernel,
        "limit impulse is no longer an accumulated projection")
require(kernel.index("Final equality evidence includes paired limit corrections.") >
        kernel.index("mrNumiHumanProjectEqualityLimitBlock("),
        "equality diagnostics precede final coupled corrections")
needle = "preloadedGeneralizedForce[dof] ="
require(runner.count(needle) == 1, "permanent preload owner is ambiguous")
expression = runner.split(needle, 1)[1].split(";", 1)[0]
require(expression.strip() == "static_cast<float>(runtimePassiveForce)",
        "permanent Human preload is not exclusively the scoped passive reference")
require("persistent_dynamic_force_audit=" not in runner and
        runner.count("persistent_initial_force_reference=") == 1,
        "static reference reactions are mislabelled as actual dynamic force evidence")
print("Human constraint-owner structural checks passed")
