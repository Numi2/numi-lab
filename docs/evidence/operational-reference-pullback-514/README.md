# Operational reference pullback and registered-volume import

This increment adds a fail-closed proof for an explicit same-topology operational reference update, with an actual CPU-only registered-volume import control. The update keeps the immutable 330 mass/source ledger and separately certifies source geometry 002 to operational candidate 514. It is a preparation/import result only; it is not a repaired-volume admission or motion result.

The tested chain loads the pinned parent packet, source 002 and candidate 514 TetGen data, proves the 330-to-002 source registration and the 002-to-514 same-topology pullback, authenticates the pinned O/I/D sidecars, imports them through the existing registered-volume importer, validates the 157 body pose rows and 86 native frame pins, and runs the existing full86 role consumer and typed O direct adapter. The run reports 7,233 volume nodes, 2,411 rows in each O/I/D role, 4,777 faces per role, 2,411 shared I rows and 14 D attachments.

The pullback tests reject a changed node map, stable face identity and immutable source mass. The importer run remains explicitly unadmitted: 183 geometry/bone pairs are present. Runtime creation, GPU work and accepted physical steps are all zero.

The 330-to-002 source registration residual is 0.351375 mm and passes its 0.4 mm source-registration gate. The separate 002-to-514 operational update is 0.218289 mm. The 0.484562 mm historical 330-parent registration reported by the 514 sidecars remains visible as historical provenance; it is not used to qualify the operational update. The immutable 330 source mass remains 13.939678963460464 kg, with per-material/geometric-donor/region closure checked by the actual pullback.

The new helper and focused test are not yet production-routed. The source-only CPU importer does not exercise the application Runtime, GPU renderer, contact solve, accepted motion or geometry admission.
