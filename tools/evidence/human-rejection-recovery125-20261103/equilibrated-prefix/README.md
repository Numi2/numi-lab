# Row-equilibrated selector6 prefix ICB

The existing opt-in prefix graph now admits the base selector6 row-equilibrated solve and captures the same physical S^-1*v scratch and offset as direct preconditioning. Cache identity includes input buffer, offset and scaling mode; Metal indirect-resource declarations include the captured input. Buffer barriers and mandatory unscaled true-residual certification remain. Partitioned, flat-column and aggregate restrictions are unchanged. No equations, precision, physical model or acceptance thresholds change.

The native row-equilibration regression passes304 checks, including repeated scratch updates and completed-solve latching. Existing scaling and raw physical-residual checks remain. Native build passes.

Matched four-step full-Human control031 versus candidate035: four accepted15.625us steps each, actual row_equilibrated_prefix=1, all833,232,924 endpoint capture bytes identical (SHA25666c476e14b867eb9a19039ec112750a3d517a18ce039b24bdb5181fec7ac24c7), physiology CSV byte-identical. Mean GPU13.5625477→13.5457279s:0.124% lower in one pair, too small to claim a performance improvement. Inclusive elapsed is slightly worse. This proves the scheduling path's integrated parity over the tested interval, not sustained performance.

Diagnostic Krylov16 remains a screen, not a replacement for the64-budget configuration. GPU-only RTF1.15350e-6. Unbalanced release, supported rest, sustained motion, three matched pairs and>=0.1x remain unqualified. Full native raw captures remain at their pinned local paths; the parity receipt covers their entire begin-through-end bytes independently of the published prefixes.
