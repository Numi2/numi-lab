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
               "mrNumiHumanProjectIntervalImpulse(", "mrNumiHumanBilateralSolve("):
    require(symbol in kernel, f"production standing owner lost {symbol}")
require("effort += supportForce" not in kernel,
        "support reaction reintroduced as a permanent effort")
require("lambdas[3u * contact + 0u] = seed;" in kernel,
        "normal warm-start impulse is not registered")
require("limitAccumulatedImpulses[limit] = nextImpulse;" in kernel,
        "limit impulse is no longer an accumulated projection")
require(kernel.index("Final equality evidence includes full-block contact and limit corrections.") >
        kernel.index("equalityLambdas[ei] = fma(impulse,"),
        "equality diagnostics precede final coupled corrections")
require("limitEqualityIndices" not in kernel and "limitEqualityCorrections" in kernel,
        "limit solve regressed to an isolated equality pair")
require("(nv + 3u * dispatch.supportContactCount) * equalityCount" in kernel,
        "conditioned limit multipliers are missing from the response stride")
require("preloadedGeneralizedForce[dof] =" not in runner,
        "runner reintroduced a frozen passive or constraint force")
require(".passiveJointProgram = passiveJointProgram" in runner and
        "compileNumiHumanPassiveJointProgram(" in runner,
        "runner no longer transports the source passive stiffness")
require("mrNumiHumanPassiveImplicitBias(" in kernel and
        "mrNumiHumanPassiveEffectiveInertia(" in kernel,
        "passive force and effective tangent no longer share the dynamics owner")
require("persistent_dynamic_force_audit=" not in runner and
        runner.count("persistent_initial_force_reference=") == 1,
        "static reference reactions are mislabelled as actual dynamic force evidence")
print("Human constraint-owner structural checks passed")

require("contactEqualityCorrections" in kernel and
        kernel.count("conditionBilateralResponse(response, equalityCorrection,") == 2,
        "contact and limit families must share complete bilateral conditioning")
require(kernel.index("mrNumiHumanBilateralFactor(") <
        kernel.index("mrNumiHumanSupportSeedImpulse("),
        "bilateral factor must precede the support warm start")
