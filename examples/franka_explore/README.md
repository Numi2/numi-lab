# Franka directional exploration

This capability uses the joint targets and minimum-jerk timing from Franka
Robotics' official `pylibfranka` joint-impedance example at libfranka revision
`85912fe02258d8cb811d3eff1f11e52ce89e3217`. The sequence is home, forward,
right, left, and home, with a three-second transition and half-second dwell.

The Apple-native path executes the trajectory through NumiSolver's implicit
position drives and emits exact replay plus physical end-effector evidence:

```sh
numi franka-explore
```

`libfranka` is the real-robot transport, state, model, limit, and control
boundary. It is deliberately built separately on a supported real-time Ubuntu
host, because the main MetalRobo project targets Apple Silicon macOS:

```sh
cmake -S examples/franka_explore/libfranka \
  -B build/franka-libfranka -DCMAKE_BUILD_TYPE=Release
cmake --build build/franka-libfranka -j
```

Hardware motion remains disarmed unless every explicit owner acknowledgement
is present:

```sh
build/franka-libfranka/numi_franka_libfranka_explore \
  --robot-ip ROBOT_IP \
  --arm-hardware \
  --confirm-free-space \
  --confirm-user-stop
```

The bridge explicitly enables libfranka's command rate limiter and 100 Hz
low-pass filter. It does not perform automatic error recovery, silently weaken
collision thresholds, or infer that a simulator run is hardware evidence.
