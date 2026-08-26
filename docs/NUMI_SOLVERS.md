# Numi solver configuration

Numi Lab exposes solver choice as a provenance-bearing configuration surface,
not as one global enum that falsely makes unrelated algorithms interchangeable.
The catalog distinguishes contact, dynamics, generalized constraints,
integration, continuum, and rod domains. It includes every independently
selectable or externally configurable solver family in the public runtime.
Internal line searches, preconditioner passes, factorization fallbacks, and
constitutive local iterations remain parts of their owning solver unless they
have an independent public configuration and publication contract.

## Discover and inspect

```sh
numi solvers list
numi solvers list --domain contact
numi solvers list --target 'CompiledRun task rollout'
numi solvers inspect contact.temporal-cone-metal
numi solvers paths
```

Each descriptor names its live implementation, public configuration owner,
documentation, selection targets, typed parameters and defaults, backend,
precision, exact selector, role, availability, and evidence boundary. `production`, `quality`,
`reference`, and `component` describe how an implementation is exposed; they
are not performance or physical-validity rankings.

## Solver profiles

Create an immutable profile from the currently resolved descriptor:

```sh
numi solvers configure contact.temporal-cone-metal \
  --profile contact-production \
  --set velocity_iterations=4 \
  --set streamed_articulated_contact_responses=true

numi solvers show contact-production
```

The profile defaults to `.numi/profiles/solvers/NAME.json` and records the
descriptor source and canonical SHA-256. `--scope user` writes beneath the user
Numi configuration instead. Existing profiles are never overwritten; make a
new profile when the solver or its parameters change. `show` fails closed when
the winning descriptor's fingerprint has drifted.

A profile records a validated configuration and its intended owner target. It
does not by itself inject arbitrary parameters into a runtime. Codex must use
the descriptor's `selection.kind` and `selection.targets` to configure the
lowest owning API or capability. A reference/component solver that lacks a
task-rollout target must not be presented as a `numi train` backend.

## External solver overlays

An external solver uses the same `numi.solver.v1` descriptor schema. Validate
and register it without changing the dispatcher:

```sh
numi solvers validate /path/to/acme-contact.json
numi solvers register /path/to/acme-contact.json --scope workspace
numi solvers inspect contact.acme
```

A minimal descriptor is data-only and must state the adapter boundary rather
than claiming a runtime it does not own:

```json
{
  "schema": "numi.solver.v1",
  "id": "contact.acme",
  "name": "Acme contact solver",
  "domain": "contact",
  "summary": "Externally owned contact solver.",
  "backend": "external",
  "precision": "owner-declared",
  "role": "experimental",
  "availability": "external",
  "owner": {
    "implementation": "/opt/acme/solver",
    "configuration": "AcmeConfig",
    "documentation": "/opt/acme/README.md"
  },
  "selection": {
    "kind": "external-adapter-required",
    "selector": "AcmeConfig.algorithm=contact",
    "targets": ["Acme API"]
  },
  "parameters": {
    "iterations": {"type": "integer", "default": 16, "minimum": 1}
  },
  "evidence_boundary": [
    "Registration validates configuration data and does not execute the solver."
  ]
}
```

Resolution follows the normal Numi ownership model, first match by solver ID:

1. `<workspace>/.numi/solvers/*.json`
2. directories in `NUMI_SOLVER_PATH`
3. `${XDG_CONFIG_HOME:-~/.config}/numi/solvers/*.json`
4. the bundled catalog

External registration copies a validated descriptor and refuses to overwrite
an existing entry. An overlay may intentionally replace a bundled ID, but its
new fingerprint makes old profiles stale. External code is not executed by
catalog discovery, validation, registration, or profile creation.

## Evidence boundary

Solver configuration is source and provenance evidence. Qualification still
requires the owning executable and, according to the claim, exact replay,
typed failures, physical outcomes, profiler counters, retained/peak memory,
and matched workload/device measurements. Changing a solver, tolerance,
iteration budget, warm start, integration method, or response path can change
the run fingerprint and requires a new comparison. Simulator evidence remains
distinct from hardware and calibrated-material evidence.
