"""Batched form of NumiHumanCoarseTissuePreparation's signed checkPrism.

No cell reorientation, absolute-value admission or thickness repair is performed.
Use Float32 determinant magnitudes only AFTER signed-prism admission.
"""
import numpy as np


def signed_prisms(outer, inner, thickness, faces):
    faces = np.asarray(faces, dtype=np.int64)
    sorted_faces = np.sort(faces, axis=1)
    a, b, c = sorted_faces.T
    odd = ((faces[:, 0] > faces[:, 1]).astype(np.int8)
           + (faces[:, 0] > faces[:, 2]).astype(np.int8)
           + (faces[:, 1] > faces[:, 2]).astype(np.int8)) % 2 != 0
    o, i = np.asarray(outer, dtype=np.float64), np.asarray(inner, dtype=np.float64)
    cross = np.cross(o[b] - o[a], o[c] - o[a])
    sg = np.einsum('ij,ij->i', cross, i[c] - o[c])
    v0 = sg / 6.0
    v1 = np.einsum('ij,ij->i', np.cross(o[b] - o[a], i[b] - o[a]), i[c] - o[a]) / 6.0
    v2 = np.einsum('ij,ij->i', np.cross(i[a] - o[a], i[b] - o[a]), i[c] - o[a]) / 6.0
    values = np.column_stack((v0, v1, v2, sg))
    tol = np.maximum(1e-8, np.max(np.asarray(thickness)[sorted_faces], axis=1)) ** 3 * 1e-12
    bad = ((~np.isfinite(values).all(axis=1))
           | (np.min(np.abs(values), axis=1) <= tol)
           | ((v0 > 0) != odd) | ((v1 > 0) == odd)
           | ((v2 > 0) != odd) | ((sg > 0) != odd))
    return bad, values


def float32_tet_determinants(outer, inner, faces):
    """Return signed normalized 3-tet determinants and volumes per face.

    Uses emitted Float32 positions and the native sorted-global-ID template.
    Signed-prism admission must precede accepting any absolute magnitude.
    """
    n = len(outer)
    a, b, c = np.sort(np.asarray(faces, dtype=np.int64), axis=1).T
    ids = np.stack((np.column_stack((a,b,c,n+c)),
                    np.column_stack((a,b,n+b,n+c)),
                    np.column_stack((a,n+a,n+b,n+c))), axis=1)
    xyz = np.concatenate((outer, inner), axis=0).astype(np.float32).astype(np.float64)
    p = xyz[ids]
    longest2 = np.zeros(ids.shape[:2], dtype=np.float64)
    for j in range(4):
        for k in range(j + 1, 4):
            d = p[:,:,j] - p[:,:,k]
            longest2 = np.maximum(longest2, np.einsum('...i,...i->...', d, d))
    determinant = np.einsum('...i,...i->...', p[:,:,1] - p[:,:,0],
                            np.cross(p[:,:,2] - p[:,:,0], p[:,:,3] - p[:,:,0]))
    return determinant / np.maximum(longest2 ** 1.5, 1e-30), determinant / 6.0
