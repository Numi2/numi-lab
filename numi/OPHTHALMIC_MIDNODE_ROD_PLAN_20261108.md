# Free-middle-node axial rod: independent discrete reference

The all-pinned examples expose false reaction convergence, but do not show how the same algorithm behaves when a rod node can move. Before designing a general repair, test a two-edge, three-node straight rod with hard endpoints and one moving middle node, without contact, gravity, damping, bend or twist motion. Use the existing effective axial parameters and a 1 ms step; this is a numerical fixture, not a suture qualification.

For an ordered one-dimensional configuration with edge stiffnesses k_i = EA / L_i, predicted middle position y = x + h v, endpoint positions a and b, and middle mass m, the independent FP64 implicit-Euler discrete minimizer is

x_new = (m y / h^2 + k_0(a + L_0) + k_1(b - L_1)) / (m / h^2 + k_0 + k_1).

Its endpoint reactions are +k_0(x_new-a-L_0) and -k_1(b-x_new-L_1), because endpoints are initially at their targets with zero velocity. Check that both solved edges remain positive, making the one-dimensional formula applicable. Evaluate symmetric prestrain, asymmetric starting position, and nonzero middle velocity separately. Require accepted output, each endpoint reaction within 1% of its reference with a 1e-6 N absolute floor, middle displacement within 1 micrometre and middle velocity within 0.001 m/s. A rejection fails completion but is not an incorrect accepted force. Preserve every result.

Use the existing Metal owner with 128 iterations, 1e-10 m correction tolerance and the same material/model inputs. Run unchanged production Metal and the existing isolated two-node special-case candidate. Their current difference should not affect three-node execution. This is a one-step, one-dimensional axial diagnostic, not a repair or a claim about bending, contact or surgery.
