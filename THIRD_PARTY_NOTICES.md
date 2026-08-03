# Third-party notices

MetalRobo's engine implementation is original work. Pinned robot and task
model constants are adapted from the sources identified below. These notices
do not imply endorsement by Franka Robotics, Unitree Robotics, the Isaac Lab
Project, the ORBIT-Surgical Project, JHU, Intuitive Surgical, Medtronic, or
their contributors.

## Franka model data

The Franka kinematic, inertial, and limit constants in `src/core/Franka.cpp`
are adapted from `frankarobotics/franka_description`, tag `2.8.1`, commit
`02afaae282d4a8e10d7d2f781b23b3515c303ce5`.

That source is licensed under the Apache License, Version 2.0. The upstream
license text is retained by reference at
<https://github.com/frankarobotics/franka_description/blob/2.8.1/LICENSE>.
The documentation image `docs/media/metalrobo-franka-fr3v2.webp` is a
MetalRobo render of the official FR3v2 and Franka hand visual meshes from that
revision. The primitive collision spheres and displayed workcell are
MetalRobo-authored. Upstream mesh files and the transient cooked pack are not
redistributed here.

## Unitree G1 model data

Source: `unitreerobotics/unitree_ros`, commit
`aa0f5c68b5aba347bad409e71b6430407da758d7`, including
`robots/g1_description/g1_29dof_rev_1_0.urdf` and its companion MJCF.
The documentation images `docs/media/metalrobo-unitree-g1.webp`,
`docs/media/metalrobo-sensor-gallery.webp`,
`docs/media/numi-lab-g1-native-rollout.gif`, and
`docs/media/numi-lab-g1-sensor-rollout.gif` are MetalRobo renders of the
official visual meshes referenced by that URDF. The upstream mesh files and
transient cooked packs are not redistributed.

BSD 3-Clause License

Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## HumanUP supine G1 reset seed

The canonical 23-DoF supine pose adapted into the bundled
`unitree_g1_supine_get_up_discovery` TaskPack comes from
`RunpeiDong/HumanUP`, commit `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`,
specifically `simulation/legged_gym/legged_gym/envs/g1waist/g1waist_up.py`.
MetalRobo maps that pose to its official 29-DoF G1 ordering; no HumanUP code,
policy weights, or assets are redistributed.

SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.

SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin. All
rights reserved.

The upstream file was modified by HumanUP authors in 2024-2025 and is offered
under the BSD 3-Clause License:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## PAC-MAN perceptive dodge training method

The native projectile-dodge TaskPack design was informed by
`lzyang2000/perceptive_cbf_rl` (PAC-MAN), including its perception-conditioned
actor, privileged per-link clearance shaping, intermittent throws, and
ball-free standing anchors. Its public deployment contract also informed the
16x9 ball-only masked-depth input, 0.1--5.0 m normalization, and sparse frame
offsets 0, 3, 8, and 18. MetalRobo implements these ideas independently in its
generic TaskIR and Metal runtime; no MuJoCo Warp, PyTorch, or policy runtime is
redistributed. The compact `assets/motions/pacman-g1-dodge.motionpack`
contains anchor-relative tracked-link pose features converted from the seven
upstream `Dodge` NPZ clips at commit
`2d426697dfc92bf1fc270f89ca92a0476033243b`.

MIT License

Copyright (c) 2026 Lizhi Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Studio Small 03 HDR environment

The README renders use Greg Zaal's
[`Studio Small 03`](https://polyhaven.com/a/studio_small_03) HDRI from Poly
Haven. Poly Haven publishes the asset under the Creative Commons CC0 1.0
Universal public-domain dedication. The source HDR and transient cooked
environment pack are not redistributed here.

## Named Unitree RL Lab training preset

The optional named PD, armature, and reset preset in `src/core/G1.cpp` is
adapted from
`unitreerobotics/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
at commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`. That source file carries:

Copyright (c) 2022-2025, The Isaac Lab Project Developers.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause.

The BSD 3-Clause terms reproduced above apply to that per-file source. The
upstream repository also contains an Apache-2.0 `LICENCE`. MetalRobo keeps the
training preset named and separate from the physical Unitree asset so it is
not presented as hardware truth.

## ORBIT-Surgical PSM model data

The topology, body masses, reset state, and named actuator preset in
`src/core/SurgicalPSM.cpp` are adapted from ORBIT-Surgical commit
`6e47534f7d412e4be523116f250c992a63146883`, specifically `psm_col.usd` and
`orbit/surgical/assets/psm.py`. The fixed 0.1 kg tooltip mass is folded into
its moving yaw parent; its combined inertia is an explicitly documented
MetalRobo approximation because the USD does not author those tensors. No
upstream mesh is redistributed.

Copyright (c) 2024, The ORBIT-Surgical Project Developers.

All rights reserved.

SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The JHU dVRK Classic PSM and Large Needle Driver definitions at
`sawIntuitiveResearchKit` commit
`53a401d014e5ef8a7d5e3ad05f0680084507662c` supply the authoritative modified
DH geometry, position and effort limits, Classic shaft/wrist dimensions, and
the Large Needle Driver 400006 4x4 `ActuatorToJointPosition` transmission used
by the serial model and its hardware-facing command map. Their repository
points to the CISST Software License Agreement
at
<https://github.com/jhu-cisst/cisst/blob/7e95680b9461009b745567f382d1b498eabc046b/license.txt>.
The complete agreement and required attribution preface are retained in
`licenses/CISST_LICENSE.txt`. MetalRobo does not redistribute JHU source or
mesh assets.

## GS-21 product facts

The procedural needle uses the Medtronic GS-21 catalog facts “37 mm,”
“half-circle,” and “taper.” All cross-section, density, tip/swage profile,
contact, and grasp-zone values are separately labelled MetalRobo research
defaults. No Medtronic mesh, artwork, documentation, or software is
redistributed.

## Sharpa Wave hand and tactile assets

`include/metalrobo/Wave.hpp` and `src/core/Wave.cpp` load, validate, and cook
external assets from:

- `sharpa-robotics/sharpa-urdf-usd-xml`, commit
  `6eea427eb24189519f32b9f21674cd534d3f973c`
- `sharpa-robotics/sharpa-tactile-sensor-assets`, commit
  `865530a98a0ca0e69d177f2121833f8bb3ed94de`

Both upstream repositories identify the work as Copyright 2025 Sharpa Group
and license it under the Apache License, Version 2.0. MetalRobo does not
redistribute the URDF, mesh, NumPy, OBJ, USD, or XML assets; callers provide
their own pinned checkouts. The Apache 2.0 terms are available at
<https://www.apache.org/licenses/LICENSE-2.0>.

The optional `SharpaIT/Robotic_Origami_Challenge` training dataset is pinned
separately at commit `8194af6b9341dac7686c2f29704ff893e6f2f95e`. Its
Hugging Face card declares Creative Commons Attribution 4.0 International
(CC-BY-4.0). Dataset files are gated, downloaded by the user, and are not
redistributed with MetalRobo. The license terms are available at
<https://creativecommons.org/licenses/by/4.0/>.
