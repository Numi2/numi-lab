# Native replay writer round-trip correction

A retained native recording was rejected by BirdFlow because the writer hashed in-memory Foundation numbers while the reader hashed the parsed JSON; the embedded digest could not be independently verified from that file. A byte-tracked normalization can render that old file but does not retroactively verify its embedded digest.

An isolated writer copy serializes and parses the payload before hashing and writing it. Matched 50-step native runs with the original and revised writers had identical parsed payloads, including every recorded q, v, rigid-body state and action. The strict BirdFlow loader rejected the baseline file and accepted the revised file; it found both recorded toe segments on all eight digits. Commands, input hashes, loader logs and replays are retained. The shared executable and source were not modified. This is a short developmental preflight, not a full-journey or physics qualification.
