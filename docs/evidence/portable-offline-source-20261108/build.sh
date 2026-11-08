#!/bin/bash
set -euo pipefail
src=/Users/numi/numi-minimal-source-restored-20261108/source
build=/Users/numi/numi-minimal-source-offline-20261108/build
prefix=/Users/numi/numi-minimal-source-offline-20261108/installed
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
[ ! -e "$build" ] && [ ! -e "$prefix" ]
cmake -S "$src" -B "$build" -G Ninja -DBUILD_TESTING=ON > /Users/numi/numi-minimal-source-offline-20261108/configure.log 2>&1
cmake --build "$build" -j 4 --target metalrobo metalrobo_bench metalrobo_world_pack metalrobo_episode_compile metalrobo_motion_compile metalrobo_visual_cook metalrobo_environment_cook metalrobo_robot_catalog metalrobo_neuron_culture_probe metalrobo_numi_window_catalog_compile metalrobo_booster_k1_compile NumiNeuronLab metalrobo_task_rollout metalrobo_task_train metalrobo_numi_window_check metalrobo_numi_policy_status_check numi_matter_runtime numi_matter_compiler numi-matterc numi-matter-physiologyc numi-matter-wet-lab > /Users/numi/numi-minimal-source-offline-20261108/build.log 2>&1
cmake --install "$build" --prefix "$prefix" > /Users/numi/numi-minimal-source-offline-20261108/install.log 2>&1
