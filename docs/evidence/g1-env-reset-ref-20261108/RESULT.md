# Result: changed-frame reset tracker gate failed at actor comparison

Preregistered at 115439ce. UTC process window 2026-11-07 23:42:00.599664–23:42:01.114755; exit 2. Preflight/postflight pinned input hashes passed. All output and exact partial snapshots are retained.

Four nonterminal warmup steps succeeded; step 5 produced env-0 world/contact status 5 (9 required contacts) and env-1 status 0. The ordinary divergence guard refused the next submission. The isolated one-step override then saw the actual task reset mask (1,0), successful statuses, and the declared status-gated clock records. Env-0's reset sample differed from its held frame in 9 depth words and 3 accepted-metadata words. A separately compiled tracker instance receiving the captured reset-time bodies and actual reset mask produced all 48 env-0 private tracker words bit-for-bit equal to the continued owner.

The preregistered actor comparison failed: 41 of 42 bound actor-history words differed. Inspection identifies a key confound: the instrument captures the owner's actor history in its finish hook after encodeTaskComplete, whose writeFrame rewrites actor history from the post-step task state. The separate tracker reference captured its output immediately after observation. These are different lifecycle points, so the mismatch does not establish a tracker reset defect. Nor does the private tracker match establish correct policy-visible actor publication. Frame/reference checks later in the probe were not reached; no conclusion is claimed for those checks. The next separate fixture must capture the actor buffer immediately after device observation and before policy inference, and compare equivalent points. Preserve this failed test unchanged.

The one-attempt override and manually supplied reset clock are research only. The unchanged production divergence guard remains enabled. No no-reset retry, full reset transaction, masked-depth history or deployment qualification is established.

