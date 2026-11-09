# Offline correction: promising, not a native result

Using the retained n=8, iteration-3 stage and proposed correction, an
independent FP64 evaluation of the same discrete mechanical equations gives a
full-step momentum norm of 1.98641e-13 N s at the retained stage and
6.30409e-15 N s when its FP32 correction is added without losing its low bits.
Both offline constructed endpoints have positive clearance (about 0.197 mm),
and their maximum kinematic defects are below 3.6e-21 m. The existing full-step
limit is 1.37215e-13 N s. This is one noncontact counterfactual correction,
not a simulated accepted transaction, contact response, convergence proof,
performance measurement, or guarantee of a passing later step. The extracted
inputs can be reconstructed from the retained rich trace with the included
script. The input capture and trace hashes, diagnostic source and executable
hashes, and full numerical output are retained.

A native extension requires authoritative stage precision through every
Newton correction, line-search evaluation, residual and tangent application,
kinematics, force and clearance calculation, and rejection/restore boundary.
Adding a low word only to the endpoint has already failed. The mode remains
explicitly disabled and unqualified while those contracts are examined.
