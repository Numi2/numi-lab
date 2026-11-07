# Result: setup rejected before physics

The preregistered process ran 2026-11-07 22:53:40.258942–22:53:40.912082 UTC and exited 2. Both arms were constructed and reported matching world, task, observation and action fingerprints, but the first control advance failed with `invalid_dimensions`: the device observation program requires a complete callback, native task, and native policy. The fixture used external zero actions without installing a native policy. There are no physics steps or sensor output rows, and no conclusion about mixed-batch failure, tracker state or restoration. The hashes and complete streams are retained beside this report.

The owner source explicitly enforces this requirement in MetalWorld.mm. A separate follow-up must install a correctly compiled, contract-matched constant-output native PolicyPack and request a new GPU slot; old delayed-action policies must remain rejected. No code or acceptance gate was changed to bypass this owner contract.
