#!/bin/bash
set -euo pipefail
base=/Users/numi/numi-verified-final-offline-20261108
archive=/Users/numi/numi-verified-final-bundle-20261108.tar.gz
sha=$(awk '{print $1}' /Users/numi/numi-verified-final-bundle-20261108.sha256)
rev=a182448ec9d987ef9446fea384560d61cb5ed4ee
src="$base/extracted/source"
build="$base/build"
prefix="$base/installed"
export PYTHONDONTWRITEBYTECODE=1 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
[ ! -e "$base/extracted" ] && [ ! -e "$build" ] && [ ! -e "$prefix" ]
printf '%s  %s\n' "$sha" "$archive" | shasum -a 256 -c - > "$base/archive-check.txt"
mkdir "$base/extracted"
tar -C "$base/extracted" -xzf "$archive"
cmake -S "$src" -B "$build" -G Ninja -DBUILD_TESTING=ON > "$base/configure.log" 2>&1
cmake --build "$build" -j 4 --target metalrobo metalrobo_bench metalrobo_world_pack metalrobo_episode_compile metalrobo_motion_compile metalrobo_visual_cook metalrobo_environment_cook metalrobo_robot_catalog metalrobo_neuron_culture_probe metalrobo_numi_window_catalog_compile metalrobo_booster_k1_compile NumiNeuronLab metalrobo_task_rollout metalrobo_task_train metalrobo_numi_window_check metalrobo_numi_policy_status_check numi_matter_runtime numi_matter_compiler numi-matterc numi-matter-physiologyc numi-matter-wet-lab > "$base/build.log" 2>&1
cmake --install "$build" --prefix "$prefix" > "$base/install.log" 2>&1
ctest --test-dir "$build" -R '^(numi_source_provenance|numi_neuron_installed_provenance|metalrobo_numi_policy_status)$' --output-on-failure > "$base/ctest.txt" 2>&1
python3 "$src/python/metalrobo/numi_source_provenance.py" stamp --archive "$archive" --source "$src" --prefix "$prefix" --expected-revision "$rev" --expected-archive-sha256 "$sha" > "$base/stamp.json"
"$prefix/libexec/numi/numi_source_provenance.py" audit --prefix "$prefix" --expected-revision "$rev" --expected-archive-sha256 "$sha" > "$base/audit.json"
cat "$prefix/share/numi/REVISION" > "$base/revision.txt"
