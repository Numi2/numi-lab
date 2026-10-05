# Retained failed preregistration attempt

The first v2 preparation attempt stopped before writing a plan or running any simulation. It found that the shared build's current `HumanRespiration.metallib` had changed since the frozen Dense45 endurance executable was recorded: the executable receipt expects `079714a5efe672ab5e5040814e434e996188c2eaeed0bffee8392cb558dc312b`, while the shared build path then contained `f1fe31119fb00f310d3f64d42b6a6afa6c8e7183971e88e64b21c1b9a2eb72d3`.

That mismatch correctly prevented registering or launching the paired study against an ambiguous runtime. The formal study is being rebuilt into a dedicated Mini build directory with private embedded library paths; this attempt remains as a provenance record and is not a registered study.
