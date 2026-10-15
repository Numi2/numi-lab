# Selected native graph assembly audit

Native run212 is an intentionally one-Newton instrument with zero accepted full controls. This audit reads its actual compiled.nmatterpack with the same frozen068 compiler package reader. The raw static tetrahedra supply exact FP32 inverse rest matrices, volumes, connectivity and material frames; rounded authored geometry is not used to reconstruct those quantities. The exported world identity equals the captured native world identity. Captured candidate masses/fixed tags match the package, and every selected cached tangent is valid.

An independent unsymmetrized FP64 finite-element assembly applies the captured local tangent, material-frame transform, force sign, timestep squared and exact masses. It compares every actual native patch LU (with its complete stored row permutation) to the principal submatrix. All13 patches pass the1e-3 matrix gate; maximum relative Frobenius discrepancy is4.724e-7, including native matrix-accumulation and LU roundoff. Same-generation assembled A*P agrees to9.899e-8, below1%.

This establishes selected assembly consistency, not a material-derivative oracle, local inverse accuracy, whole native solve, mesh convergence or shipping-box qualification. The ill-conditioned local solve remains separately failed in native210. Unfactored patch matrices were not retained, so accumulation and LU roundoff are not individually separated here.

The root script is /Users/home/cardboard-implementation-20261014/audit_native212_patch_assembly.py. export.cpp and its compiled helper run only on CPU; raw arrays and report.json retain all source/package/payload/compiler hashes. The complete native failing transaction and certificates remain under solver/graph-schwarz-topology-factor-one-newton-212.
