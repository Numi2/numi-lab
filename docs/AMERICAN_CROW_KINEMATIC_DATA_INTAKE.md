# American-crow kinematic data intake

This is an unsent request and acceptance specification for a future licensed
American-crow flight-data handoff. It does not authorize acquisition, data
sharing, use of animal data, or a new flight claim by itself.

## Purpose and scientific boundary

The target use is to identify a clearly labelled Numi research model for
standing-to-flight control. The nearest published experiment recorded vertical
escape takeoff, not unassisted forward flight; it used three synchronized
250 Hz cameras and digitized 3D wing/body markers. See the primary
[Corvidae escape-flight study](https://journals.biologists.com/jeb/article/214/3/452/33507/Scaling-of-mechanical-power-output-during-burst)
and its source [dissertation](https://scholarworks.umt.edu/etd/960/).

Published figures, movies, and species-level summary tables are literature
evidence only. They must not be digitized into a claimed specimen trajectory or
combined silently with the independent BirdFlow visual surface.

## Requested package

Request the original records, with a written license covering archival,
derivative-model, internal research, and publication/showcase use:

1. Raw synchronized camera video for each usable trial, plus frame rate,
   frame count, shutter/exposure metadata, camera identifiers, and recorded
   synchronization pulses.
2. Camera intrinsics, distortion, extrinsics, calibration-target definition,
   world axes, units, and the reconstruction/reprojection method.
3. Per-frame 3D marker coordinates with marker names, left/right labels,
   visibility/occlusion flags, filtering/resampling history, toe-off,
   touchdown, and wingbeat-event annotations.
4. Trial metadata: specimen pseudonymous ID, date, task/chamber geometry,
   start pose, trial inclusion/exclusion reason, atmosphere, and any forceplate
   or external-load signals synchronized to the camera timebase.
5. Same-specimen morphology: body mass, center of mass and its frame,
   full-body inertia tensor, wing masses, hinge-to-COM vectors, segment
   inertias, shoulder/wing geometry, and the measurement method and
   uncertainty for each.
6. A provenance manifest naming every measurement, transformation, instrument,
   investigator, license restriction, and cryptographic file hash.

If the data are a multi-specimen or multi-source composite, retain each source
identity and build an explicit hybrid record. Do not create a fictional
"same-specimen" package.

## Intake quality gates

Before the data influence a compiler, controller, or reward:

- Verify hashes and license scope; retain original bytes read-only.
- Reconstruct a trial and report reprojection error, timing residuals, dropped
  frames, occlusions, coordinate handedness, units, and all filters.
- Check bilateral labels and anatomy against the registered visual/mechanical
  model; record every registration transform.
- Keep raw coordinates, derived joint angles, surfaces, model parameters, and
  controller targets as separate versioned artifacts.
- Hold out at least one whole trial and one wingbeat segment before fitting any
  kinematic or aerodynamic closure.
- Compare an executable replay against held-out markers and event times before
  allowing it to seed policy learning.

Failure at any gate leaves the material as literature or exploratory evidence;
it does not create an eligible training target.

## Acceptance outcomes

| Package status | Permitted disposition |
| --- | --- |
| Complete same-specimen geometry, kinematics, inertia, and license | Create a provenance-locked measured record; qualify replay before training. |
| Complete kinematics but missing mass/inertia or reuse rights | Keep as a non-training research reference; request the missing terms. |
| Aggregate summaries, figures, or movies only | Cite as literature context; no digitization or controller fitting. |
| Deliberate multi-source hybrid with every source declared | Create a separately named estimated hybrid; never call it measured American-crow flight. |

## After acceptance

Pre-register the exact task, train/hold-out split, source hashes, model
changes, physical failure definition, selection threshold, and replay capture
before launching a learner. A README crow GIF requires both a promoted policy
and a visually inspected deterministic replay; it remains forbidden for an
unselected candidate or an incomplete data record.
