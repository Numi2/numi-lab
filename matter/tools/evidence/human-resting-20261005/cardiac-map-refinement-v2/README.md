# Ventricular map refinement

The native loader now admits source-bound corrections to the existing RV, LV and wall-closure coefficient fields. The compiler binds the actual cardiac map dependencies, retains an explicitly replaced descriptor, and packs sparse Float32 coefficients by their bytes, including signed zero. Lumen-owned vertices cannot change. This uses the existing GPU map and material-volume root; it adds no physical state or alternate simulator.

The current-main build passed 12 focused Python checks, the cardiac identity CTest, and a six-second native integrated run on the M4 Pro Mac mini. All eight recorded ID23 wall states passed exact self/degeneracy and RA/LA crossing checks. The current-main replay's five cardiac surfaces are byte-identical to the independently audited run. The physiology CSV and full-precision terminal body serialization are unchanged. Rejected-step freezing and retry replay passed.

The 95-frame recording remains at the exact path and SHA-256 in summary.json. It preserves wall-clock timestamps. The SSH session had no GUI drawable; recording used the native framebuffer.

This increment does not qualify the complete human. Atrial walls, pleura, bowel interfaces, final long-run physiology, and the five-minute continuous scene remain separate gates. The geometric evidence covers eight recorded states, not every point in a continuous deformation domain. The sampled displacement accounting and the unmet nominal optimization margin are explicit in summary.json.
