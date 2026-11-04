# Safeguarded contact Armijo interpolation

The per-contact search exhausted its fixed dyadic trials at native148 root26. The new proposer uses the measured trial energy and analytic slope to choose the next alpha inside [0.1 alpha, 0.5 alpha]. Acceptance still requires the existing exact rounded endpoint, full CCD certificate, finite energy and Armijo inequality. There is no change to contact energy147, masses, timestep, final residuals, evaluation budgets or rollback.

GPU-only direct interpolation checks pass (11 cases, including Float64 comparison, invalid-data halving, canary and a stiff quadratic that the same eight dyadic evaluations cannot accept). The complete solver contract suite passes 917,050 checks.

The complete Human scene now accepts 32/32 steps (0.0005 s simulated); prior148 accepted26 then rejected. All11 compact CSV traces are bitwise equal on their common26-step prefix. Common-prefix accepted GPU duration is217.774s versus217.616s; no speedup claim. All32 accepted GPU steps total331.855s (GPU-only factor0.00000150668x). Process wall time551.494s includes initialization and output. This remains an unbalanced release initialization prefix, not supported-rest settling, sustained motion, five-minute physiology or0.1x qualification. Full fine-FEM force/strain endpoint comparison remains outstanding.

Full source snapshot restores39/39 pinned source hashes. The exact plan, device/binary/asset pins, command, compact traces, failed compile logs and source preimage are retained. No native-viewer visual verification or video is claimed.

Primary algorithm reference: SciPy scalar_search_armijo quadratic backtracking, https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py . The implementation is native Metal and adds no dependency.
