# Source-owned shoulder, elbow, ankle and aggregate MTP actuation

Four matched selected-muscle/tendon/articulated native core comparisons pass: bilateral supraspinatus shoulder elevation, brachialis/brachioradialis elbow flexion, ankle dorsiflexion and nonsaturated long-toe-flexor MTP motion. Each condition advances 16 accepted 125-microsecond steps (2 ms), with bitwise replay, identical initial q/v, 51 retained NHEQ records, existing source joint ranges, finite state and mandatory native tendon force/moment closure. No joint target is held and root_assistance=false. The clean014 binary and dylib/source/header hashes are bound by the previously retained clean build/link receipts.

Positive loaded-minus-baseline q, right/left radians:
- shoulder: 5.01424e-6 / 5.05894e-6
- elbow: 2.90978e-5 / 2.92381e-5
- ankle: 5.08775e-3 / 6.76702e-4
- aggregate MTP: 5.84609e-4 / 4.02379e-4

The initial four-muscle MTP comparison failed its activation-increment check: source FDL_l index386 is already at activation1 in the retained initial state, so adding the requested increment saturates. This failed trial is retained. A separately declared three-muscle cohort omits386 from commanded IDs while keeping its original background state unchanged in both conditions. It passes all original checks, without relaxing the increment gate. The final MTP response equals the original response because386 received no effective increment in either experiment.

The command option is an increment on source activation, not an assertion that every baseline source muscle is inactive. `core-joint-cohorts-015-preregistered.json` uses historical baseline_activation/loaded_activation labels for those option values; the raw source-owner reports and the revised MTP preregistration state actual accepted activations. The initial source variants are retained under source015-initial-cohorts with hashes matching the first preregistration. The final wrapper only extends the existing runner and assessor cohort tables.

These measurements demonstrate short-horizon causal native core response. They do not establish natural speed, supported settling, full range of motion, ligament adequacy, deformable skin/fat/contact, or integrated physiology/performance. The foot model has one aggregate MTP/toes body per side, with no independent toe or IP-joint DOFs. Abrupt activation and small unsupported-core inertias can produce large ankle acceleration; a whole-body natural-motion claim requires the admitted tissues and support loads. There is no 0.1x integrated-throughput claim.
