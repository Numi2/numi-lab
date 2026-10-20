# Coracoclavicular complex catalog increment

This evidence binds two massless, inferred effective CC-complex rows in the registered-ligament catalog. It does not qualify a native physics run.

The warnings-as-errors CPU catalog fixture passed. The exact-invocation 044 geometry audit passed all 30 catalog rows at q overrides 36=0.10 and 74=0.10. Each row tested all 185 registered source-bone meshes; the two new straight paths had no source-triangle intersections. Their endpoint surface triangles, body-local float32 anchors, surface round-trip errors, and projection results are in actual-pose-044-final.json.

The right row uses scapula stable ID 10/source record 18/body 34 to clavicle stable ID 8/source record 16/body 29 in frame joint 33 (33→34). Its frame-local endpoints are [-1.526024, -57.940971, -36.693003] mm and [15.332877, -29.584742, -41.193045] mm; path length 33.295 mm; endpoint-normal cosines 0.3598 and 0.6654. The left row uses scapula stable ID 11/record 55/body 84 to clavicle stable ID 9/record 53/body 79 in frame joint 83 (83→84). Its frame-local endpoints are [-3.107176, -58.568598, 34.779065] mm and [17.153062, -29.165176, 43.675959] mm; length 36.799 mm; normal cosines 0.5934 and 0.6369. Each reported mesh-projection residual is under 0.001 mm because the seed is a rounded source-surface witness. The separate anatomical-region search target residuals are larger: 2.8/9.4 mm right and 4.0/7.8 mm left. Those are broad proxy fit errors, not precise enthesis locations.

The stiffness prior is the intact coracoclavicular complex value from Harris et al. (2000), 103±30 N/mm, represented as one 103,000 N/m aggregate tangent per side. The effective 10 mm² area only normalizes the stress law; it is not measured and the rows add no mass. The two ligament components are not modeled separately.

Geometry landmarks are inferred from the source shoulder model and published population anatomy: the CORB route-first sites (NHMYO2 records 224/287, site IDs 672/1093) define a coracoid-tip-region direction; the base direction uses the reported tip-to-precipice dimension as an anatomical scale cue. The clavicle proxy lies at 20% of the source clavicle long axis, between fresh-anatomic conoid/trapezoid fractions. Nearby source muscle sites were checked; none supplies a clavicular CC attachment witness. These are regional proxies, not measured subject entheses. Source meshes are open, so triangle-clearance is not proof of closed-solid nonpenetration. Dynamic shoulder-pose clearance remains unverified.

Primary sources:
- Harris RI et al. “Structural Properties of the Intact and the Reconstructed Coracoclavicular Ligament Complex.” Am J Sports Med. 2000;28(1):103–108. DOI 10.1177/03635465000280010201, PMID 10653552. https://pubmed.ncbi.nlm.nih.gov/10653552/
- Salzmann GM et al. “The Coracoidal Insertion of the Coracoclavicular Ligaments: An Anatomic Study.” Am J Sports Med. 2008;36(12):2392–2397. DOI 10.1177/0363546508322887, PMID 18755935. https://pubmed.ncbi.nlm.nih.gov/18755935/
- Rios CG, Arciero RA, Mazzocca AD. “Anatomy of the Clavicle and Coracoid Process for Reconstruction of the Coracoclavicular Ligaments.” Am J Sports Med. 2007;35(5):811–817. DOI 10.1177/0363546506297536, PMID 17293463. https://journals.sagepub.com/doi/abs/10.1177/0363546506297536

The fixture checks catalog metadata and generic mechanics invariants. This receipt does not include a native build/run or claim sustained shoulder support.
