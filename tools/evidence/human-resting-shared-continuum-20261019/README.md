# Shared Human FEM/CV scene evidence (0.5 ms)

This committed source slice composes an authored FEM world seed with the existing resting CV/body world in one `RestingRun` runtime. The source composition test is at `tests/resting_world_source_composition.mm`; its CPU-side result is retained under `217/`.

The integrated native evidence under `009/` records exactly two accepted intact body/FEM/fascia/CV steps at 0.25 ms each (0.5 ms simulated time). The accepted summary, exact invocation, build receipt, source pins, and compact coupled CSV are included. The detailed native log and the large MRV pack remain at their original retained paths; the accepted summary records the native log SHA-256.

This is a short integration result, not whole-body continuum qualification. It contains no full breath or heartbeat, external bleeding is disabled, and visual verification was not performed. The separate 224 test was pending at packaging time and is neither included nor claimed as passing.

The reusable headless launcher is `run_native.py`. It verifies the pinned executable, runtime library, authored assets, and source-loaded modules, then refuses to overwrite an existing output directory. The repository-relative production paths and required timing-header dependency are listed in `manifest.json`.
