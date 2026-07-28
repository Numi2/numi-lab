# Third-party notices

MetalRobo's engine implementation is original work. The pinned Unitree G1
compiled-model constants in `src/core/G1.cpp` are adapted from the sources
identified below. These notices do not imply endorsement by Unitree Robotics,
the Isaac Lab Project, or their contributors.

## Franka model data

The Franka kinematic, inertial, and limit constants in `src/core/Franka.cpp`
are adapted from `frankarobotics/franka_description`, tag `2.8.1`, commit
`02afaae282d4a8e10d7d2f781b23b3515c303ce5`.

That source is licensed under the Apache License, Version 2.0. The upstream
license text is retained by reference at
<https://github.com/frankarobotics/franka_description/blob/2.8.1/LICENSE>.
The primitive collision spheres are MetalRobo approximations; upstream mesh
assets are not redistributed here.

## Unitree G1 model data

Source: `unitreerobotics/unitree_ros`, commit
`aa0f5c68b5aba347bad409e71b6430407da758d7`, including
`robots/g1_description/g1_29dof_rev_1_0.urdf` and its companion MJCF.

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

## Named Unitree RL Lab training preset

The optional named PD, armature, and reset preset in `src/core/G1.cpp` is
adapted from
`unitreerobotics/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
at commit `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`. That source file carries:

Copyright (c) 2022-2025, The Isaac Lab Project Developers.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause.

The BSD 3-Clause terms reproduced above apply to that per-file source. The
upstream repository also contains an Apache-2.0 `LICENCE`; the exact
provenance and boundary are recorded in `docs/G1_SPEC.md`. MetalRobo keeps the
training preset named and separate from the physical Unitree asset so it is
not presented as hardware truth.
