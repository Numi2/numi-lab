# Haj-Ali 2009 asymmetric transfer candidates

The two CSVs and four `.nmatter` assets below are versioned constitutive model
inputs. They are not test observations or a complete reproduction of the
paper's constitutive model.

Primary source: [Haj-Ali et al. (2009), *Refined nonlinear finite element
models for corrugated fiberboard*](https://doi.org/10.1016/j.compstruct.2008.02.001),
especially Tables 1–3 and Figs. 1–2. The named grades are the paper's 205 g/m²
unbleached southern softwood kraft liner and 126 g/m² NSSC medium.

| Variant | Parameter CSV SHA-256 | Material assets |
| --- | --- | --- |
| Exact-ReLU provisional transfer | `47dc13cb3ced9dac438df21354b3236d53223eed0842fe97806d32507b890e0a` | liner `377236230f1802b10aba8c994df86d545a7694bc11885b3592275d36aa0eefe5`; medium `f5f0e1475a9b0a87c0c30b27b6ab5f26a04a92e50ed5ca9b5c775477ed8f6a7b` |
| C1-smoothed transfer | `5fccfea7ca49776cef6782f6fa8eff1876de38f643c6993c3b2829135f446e48` | liner `f5951cd7fb63684a4fdcca58457af5d769fe466cc640370be54d9a4ac4309a43`; medium `40e8bd9aa61ed946661c162c395397f958bab71306df8bec8df1140e78ae95bb` |

Both variants use tensile Hill radii derived from Table 3 and separate positive
compression corrections based on compression strengths that the paper assumes
to be about 60% of tensile strength. The out-of-plane ZD strengths are derived
using the paper's cited Beldie relation; shear strengths are assumptions. They
are not direct compression or shear measurements. Table 3 medium MD tensile
strength is 15.65 MPa, while Table 1 reports 15.7 MPa; the asset uses Table 3
and the discrepancy remains visible.

The two directional hardening scales transfer the digitized total tensile
tangents from the solid fitted lines in Figs. 1–2 through a uniaxial plastic
hardening conversion. The endpoint record is one pixel-read fit endpoint per
axis, and the 278 selected marker centers are from those same plotted curves.
The markers are calibration residuals only: they are not raw specimen traces,
do not provide specimen identities or a held-out set, and do not identify the
multiaxial hardening transfer. Conditioning and test rate are unavailable.

The exact-ReLU candidate uses a nonsmooth zero-stress compression corner. A
prior reported liner-MD return-map failure is retained at
`/Users/home/cardboard-implementation-20261014/analysis/asymmetric-transfer-calibration-20261015-001/legacy-exact-relu-corner-failure.json`.
That record is explicitly a reconstruction from the reported failure values;
the original raw solver log and full state vector were not available. It is a
failure record, not a fresh measurement or a successful material-point test.

The C1 candidate replaces `max(-s,0)` with a C1 positive-part smoothing of
width 5e-6 in normalized stress. This is a numerical regularizer, not a
physical parameter. Its value at zero is `delta/4`, so it creates a small
nonzero shift in the yield radius. The registered three-width CPU calibration
receipt is outside the material directory at
`/Users/home/cardboard-implementation-20261014/analysis/asymmetric-transfer-calibration-20261015-001/`.
Neither CPU curve calibration nor a material-point kernel test qualifies
physical folding, damage, debonding, or assembled-board behavior.
