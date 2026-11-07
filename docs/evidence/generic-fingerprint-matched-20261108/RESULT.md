# Matched Franka visual-depth fingerprint experiment

This CPU-only experiment uses the source and copied dylib from the same product owner revision 4aee60dc. It compiles the same source twice, adding an isolated observation-hash discriminator only when an authored visual-depth program exists. No product source or policy was changed.

The first matched fixture was rejected because it omitted the required complete visual-operator suffix; the source, output and identities are retained. A separately corrected fixture authored the eight required depth slots. Its original compiler produced 102-element nonvisual and 110-element visual zero-policy packs, both accepted by their original task contracts. The candidate preserved the nonvisual fingerprint and accepted that old policy. For the visual task, the actor shape remained 110 while the observation fingerprint changed from 7550321732695938159 to 4933159811587555512; the unmodified old policy was rejected with incompatibleContract. A policy explicitly rebound to the new contract compiled. This checks contract enforcement, not trained-policy quality or a real rollout.

The matched 4aee60dc owner lacks the research branch's unsegmented-depth switch; raw-mode migration is not qualified. The earlier ABI-mismatched probe abort is retained separately. The candidate discriminator is a research sketch and still requires proper ownership, serialization and integrated runtime compatibility tests before any product integration.

