# Native recording startup clocks on SSH Numi

The existing inspector rejected the retained native007 startup movie because both images identify physical time zero. The viewer first presents its initialized source state, then the GPU publishes the matching accepted pre-step state after physical acceptance; the existing coupling captures body poses and respiratory state together before dynamics. This is a startup presentation case, not an extra accepted physical step.

The inspector now keeps and counts leading zero-time repeats. Wall presentation timestamps must still increase strictly; after simulated time first advances it must also increase strictly. Frame-count agreement, original timestamps, exact frame extraction, and all other recording checks are unchanged. The movie is not filtered or retimed.

Five native Swift clock tests cover startup-only/advancing recordings, positive-time stalls, backward clocks, duplicate wall timestamps, and malformed clock arrays. The same retained movie failed before and passed afterward: two images, one startup hold, wall timestamps 0.975 and 2.368333333333333 seconds, original duration 3.7616666666666667 seconds. Receipts bind source, compiled inspector, commands and original movie/CSV hashes. This .002-second startup diagnostic does not establish anatomy, layer-tour, endurance or physiological qualification.
