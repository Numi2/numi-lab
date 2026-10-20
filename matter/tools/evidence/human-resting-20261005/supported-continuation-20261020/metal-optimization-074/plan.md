# 1,024-lane orthogonalization experiment

Compare the accepted four-step mechanics workload on SSH Numi (M6), keeping the existing input, timestep, and GPU-indirect/empty-field changes fixed. This bounded trial checks whether the M6 pipeline supports a 1,024-threadgroup reduction and whether its throughput and endpoint metrics improve relative to the width-aware 512-lane implementation. No timestep or physical equation changes are included.
