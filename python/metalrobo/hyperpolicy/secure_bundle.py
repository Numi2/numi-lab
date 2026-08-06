"""Runtime-authenticated motion-policy bundle surface."""

from __future__ import annotations

from .bundle import (
    MotionPolicyBundle as _MotionPolicyBundle,
    build_generated_motion_policy,
)


class MotionPolicyBundle(_MotionPolicyBundle):
    """Reject in-memory policy mutation after artifact authentication.

    NumPy arrays are mutable even inside frozen dataclasses. Deployment calls
    ``validate`` before execution, so recomputing both semantic fingerprints
    closes that gap without copying large reference tables.
    """

    def validate(self) -> None:
        super().validate()
        if (
            self.hyper_base.computed_fingerprint()
            != self.hyper_base.fingerprint
        ):
            raise ValueError(
                "hyper-base parameters changed after authentication"
            )
        if (
            self.motion_policy.computed_fingerprint()
            != self.motion_policy.fingerprint
        ):
            raise ValueError(
                "generated motion policy changed after authentication"
            )


__all__ = ["MotionPolicyBundle", "build_generated_motion_policy"]
