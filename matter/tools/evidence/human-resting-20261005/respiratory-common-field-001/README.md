# Source-bound common respiratory field

The existing native resting viewer now uses one spatial displacement field for
the five closed lung lobes, pleura and diaphragm. The existing MyoSim/tendon
respiratory mechanics supplies diaphragm swept volume `qD` and rib swept volume
`qR`. The field's diaphragm displacement is `qD / A`, with `A` calculated from
the exact loaded lobe triangles. Its smooth basal and attachment footprint is
an explicitly inferred reference registration, not a measured individual's
motion. Radial expansion contributes `qR`; it does not duplicate `qD`.
Reciprocal interface vertices use the same field. The GPU also checks each
lobe's actual rendered volume against its own swept-area contribution.

The source-repaired diaphragm/lobe payload was built by Human commit `44224d2`.
Its common field has effective area **0.019892791918944348 m²**. Both the anatomy
receipt and physiological muscle-area parameter must agree with the area
recomputed at load time, within a relative tolerance of 1e-6. Loading a stale
area is an error. The existing NHANATOMY/receipt and GPU buffers remain the
owners; no new runtime asset format or CPU simulation was introduced.

`diaphragm-common-map-native-001` completed 3,000 accepted 2 ms steps on the
SSH Mac mini, Apple M4 Pro: **6.000000285 simulated seconds / 61.404735875 wall
seconds = 0.097712338× real time**. All 95 displayed accepted states passed the
existing skin/bed and functional-volume checks. Maximum volume relative error
was 1.094025e-6 and minimum skin/bed gap was −0.110552 mm. The integrated
rejection test preserved body, circulation, respiratory and Brain histories;
retry matched uninterrupted replay, including an inert suffix after rejection
in the same command buffer. Runtime input hashes were unchanged after the run.

Actual GPU-rendered MRVPack captures at steps 0, 639, 2783 and 2999 remain on
the Mini beside the continuous native movie. Their compact receipts are
included here. The surface CSV contains the corresponding physical time and
functional targets; a neighboring physiological CSV row must not be substituted
for that state. The native presentation deliberately lags acceptance by one
2 ms step so rejected work cannot appear in the viewer.

The earlier `respiratory-common-basis-native-001` used the same mathematical
field on the old diaphragm geometry. Its prototype receipt stored the field
under provenance; the final loader reads `functional_bindings` instead.
Three 12-second native sensitivity runs on those earlier lobe nodes varied the
inferred attachment footprint, recomputed the source area, and changed the
mechanical parameter together. Effective areas 0.0177834–0.0206051 m² produced
528.13–542.24 mL tidal volume, 19.30–21.23 mm peak diaphragm displacement,
and 6.148–6.300 L/min event-ledger ventilation. Each window contains only one
complete breath interval. These are short parameter sensitivities, not steady
state or population validation. The later seam refinement changes effective
area by about 3.4e-5 relative; it was recomputed before the new native run.

The same source increment retains the tested 20-degree active cardiac
free-wall map and explicit cavity origins. Passive myocardial wall/lumen
interfaces are still unresolved. Exact accepted-frame checks also found
liver/diaphragm and preexisting abdominal-organ crossings. Consequently these
runs **do not qualify whole-body anatomy or finish the deliverable**. They
provide a reproducible integrated increment for the remaining interface work.

Exact commands, source revision/diff, parameters, runtime and asset SHA-256s,
compact complete traces, failed-step/replay result and timings are retained in
this directory. Large original packs and continuous movies remain under
`/Users/n/numi-human-resting-evidence-20261005/` on `ssh macmini`. All builds,
processing and simulations ran there; no execution used the Air.
