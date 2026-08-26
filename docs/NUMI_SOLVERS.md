# Numi solver configuration

Numi Lab exposes solver choice as a provenance-bearing configuration surface,
not as one global enum that falsely makes unrelated algorithms interchangeable.
The default catalog presents nine stable algorithm families. CPU/Metal
backends, precision levels, owner APIs, and runtime adapters are variants
beneath those families instead of being presented as separate algorithms.
Internal line searches, preconditioner passes, factorization fallbacks, and
constitutive local iterations remain parts of their owning solver unless they
have an independent public configuration and publication contract.

## Discover and inspect

```sh
numi solvers list
numi solvers list --domain contact
numi solvers list --target 'CompiledRun task rollout'
numi solvers inspect quality-newton
numi solvers inspect quality-newton --variant unified-metal
numi solvers list --implementations
numi solvers inspect contact.quality-newton-metal-world
numi solvers paths
```

`list` shows the nine families. `inspect FAMILY` shows its default and variants.
`list --implementations` exposes all exact backend/target records for auditing
and compatibility. Each implementation descriptor names its live owner,
documentation, targets, typed parameters, backend, precision, exact selector,
role, availability, and evidence boundary. `production`, `quality`, `reference`,
and `component` describe exposure, not performance or physical-validity ranks.

## Solver profiles

Create an immutable profile from a family. With no variant or target, Numi uses
the family's declared default:

```sh
numi solvers configure temporal-cone \
  --profile contact-production \
  --set velocity_iterations=4 \
  --set streamed_articulated_contact_responses=true

numi solvers configure quality-newton \
  --target 'MetalUnifiedQuality API' \
  --profile unified-quality

numi solvers configure symplectic-euler \
  --variant free-body \
  --profile free-body-integration

numi solvers show contact-production
```

The profile defaults to `.numi/profiles/solvers/NAME.json` and records the
family, variant, exact implementation ID, descriptor source, and canonical
SHA-256. Exact implementation IDs remain accepted by `configure` for scripts
written against the original catalog. `--scope user` writes beneath the user
Numi configuration instead. Existing profiles are never overwritten; make a
new profile when the solver or its parameters change. `show` fails closed when
the winning descriptor's fingerprint has drifted.

A profile records a validated configuration and its intended owner target. It
does not by itself inject arbitrary parameters into a runtime. Codex must use
the descriptor's `selection.kind` and `selection.targets` to configure the
lowest owning API or capability. A reference/component solver that lacks a
task-rollout target must not be presented as a `numi train` backend.

Profiles created before the family-first catalog lack family/variant metadata
and retain the earlier descriptor fingerprint. They remain readable but report
stale; inspect the family and create a new profile instead of silently rebinding
the old profile to a default variant.

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
  "family": {
    "id": "acme-contact",
    "name": "Acme contact",
    "summary": "Externally owned contact solver family.",
    "variant": "owner-default",
    "default": true
  },
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
