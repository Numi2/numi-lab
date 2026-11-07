#!/bin/sh
set -eu
prefix=$1
world=$2
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for executable in metalrobo_world_pack metalrobo_episode_compile; do
  otool -l "$prefix/bin/$executable" > "$tmp/$executable.otool"
  grep -F '@loader_path/../lib' "$tmp/$executable.otool" >/dev/null
done
"$prefix/bin/metalrobo_world_pack" --inspect "$world" > "$tmp/world.out"
grep -F 'articulations=1' "$tmp/world.out" >/dev/null
set +e
"$prefix/bin/metalrobo_episode_compile" > "$tmp/episode.out" 2>&1
result=$?
set -e
[ "$result" -eq 2 ]
grep -F 'usage: metalrobo_episode_compile' "$tmp/episode.out" >/dev/null
printf '%s\n' 'relocated WorldPack and EpisodeTwin tool checks passed'
