# Virtual Wet Lab v0.3 qualification log

First working increment: the existing specimen renderer now caches cell and gene
lookup maps and selected sparse gene buffers. Picking uses a spatial grid.
Control, predicted and observed layers share their comparison scale; residuals
use a symmetric legend explicitly labelled `predicted − observed`.
Raw control UMIs remain distinct from normalized regional predictions.

Verified locally on the retained v0.2 specimen (2,514 displayed cells): browser
load, canvas construction, cached lookup and selection index. Node checks cover
sparse zero versus missing values, buffer reuse, nearest-point picking and
shared/signed scales. This is not the larger-specimen performance qualification.
The old experimental records and biological verdicts are unchanged.
