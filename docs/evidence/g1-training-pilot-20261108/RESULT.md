# Fresh G1 pilot result

The installed source revision 3a439b63 completed 20 updates and matched
candidate/incumbent selection in 28.61 seconds wall time. The training output
reported zero failed environment steps. Both held-out arms reported zero
physics-error steps but no clean-horizon environments: the incumbent had 116
height-or-tilt terminations and the candidate 118 across eight environments.
Neither met the preregistered screen. Candidate final forward progress ranged
from 0.018 to 0.612 m, below the required 1 m for every environment. The
selector retained the incumbent; higher candidate mean final progress did not
offset increased terminations and worse tracking. Repeated termination counts
are events, not fractions of environments.

The selector labels its summary task "velocity" although both native evidence
records identify unitree_g1_legs_locomotion; the scoring/label path is under
inspection. These are idealized simulator observations, one training seed and
one evaluation seed. This does not establish learned control, deployment
validity, time-to-skill, or accuracy-matched performance. The training output
reported 5.309 s for its update loop, excluding selection and setup; wall time
covers the full command. Other task GPU experiments were paused, while CPU
work and background system activity were not controlled.

The complete run artifacts, policies, learner state, raw logs, state traces,
host process snapshots and pilot supervisor are retained in the archive named
by archive.sha256 on the measured host. The compact repository copy includes
both selector evidence records, training output, command and runtime hashes;
it is not the complete archive or a standalone executable environment. Python
3.13.16 and MLX 0.32.4 were used. The input manifest names the staged installed
package, which remains separately available on the host.
