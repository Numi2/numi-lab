# Native contact-budget64 028 — targeted contact rejection trace

Execution exited 1 after 77.283 s. The invocation requested 6 physical steps at 125 µs and accepted five coupled rows (0.625 ms) before failing; the six-step horizon did not complete. The FGMRES iteration budget was 64.

The failure matches 027 across segment start, matter code/object/index, microstep count, and all four diagnostics: segment 5, code 6, object 19, slot 2941, zero microsteps. All five accepted CSV rows are byte-identical to 027. The reused contact-armijo-024 analyzer found 24 raw root-5/microtick-0 events and 23 records containing contact slot 2941; iteration 22 is the qualifying failure with 12 alpha trials. The selected contact identity and source tuples are [19,19,0,1] and [16053,16094,2941,1]. Machine-readable details and TSV trials are retained under analysis/.

All 13 file inputs match 027. Invocation/provenance bind the run to executable 37d11b2888dcd4413cc22052f284d26b3f36bce971989c4112c1a2f524e82867, metallib 582a030a67312ee00f3358e4a48e8c7741e7d2b67584fe5d496f58f578532535, and contact.metalinc SHA 2043beee378e74894ade2254f8c3848e6e7274f75dc38891af004c275f7c1f95. The contact source was edited after 028 completed; two later source hashes are recorded separately and are not attributed to this run. No large output pack was copied.
