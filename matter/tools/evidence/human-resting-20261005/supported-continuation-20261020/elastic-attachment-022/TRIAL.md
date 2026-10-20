# Finite retaining-interface trial 022, predeclared
Question: do finite elastic retaining attachments avoid the strong local foot
compression caused by hard point ties, while maintaining real equal/opposite
three-dimensional connective forces at the same 20 source/donor identities?

Control: attachment-control-021, corrected reference mesh with the original 20
hard ties. It rejected step 19 after 18 accepted 125 us steps at foot tet 21728.
Trial: same assets, pose, mass, 28 ligament rows, 26 fascia regions, all shared
tissue owners and numerical validity floors. Replace all 20 skin/fat hard ties
with finite elastic body attachments in the existing GPU FEM/body interface.
No pins are selectively removed; there is no motion projection, added physical
mass, alternative physics owner, or relaxation of physical validity criteria.

Prior: each site has k = mu * incident deep-surface tributary area / local fat
thickness, in N/m. mu = E/[2(1+nu)] from the existing low-strain adipose material,
E=1.6 kPa and assumed nu=.45 (mu=551.724 Pa). The E source is Alkhouli et al.,
2013, DOI 10.1152/ajpendo.00111.2013, PMID 24105412
(https://pubmed.ncbi.nlm.nih.gov/24105412/), whose initial subcutaneous modulus
was 1.6 +/- .8 kPa. Isotropic, unresolved retaining-interface stiffness is an
explicit modeling inference, not a measured ligament or enthesis property.
Sensitivity endpoints .5 and 1.5 scale are declared, if the base trial passes.
The same local point has zero spring energy in the registered reference pose.

Numerical prediction: all requested 32 steps (4 ms) accepted, finite states,
unchanged total physical mass, no assistance, no Jacobian/solver-floor changes.
Every actual failure is retained. The force fixture must first pass attraction,
repulsion, tangential response, energy gradient, rigid-motion objectivity and
equal/opposite force/moment checks; old unilateral glide must stay unchanged.
An endpoint state is captured for numerical examination, with no visual check.

This transient release is not equilibrated bed support or sustained physiology
qualification. Even 32 accepted steps do not establish full-body completeness,
long-horizon validity, all anatomical clearance or human realism.
