#!/bin/sh

set -eu

output=${1:?usage: crow_policy_interpolation_screen.sh OUTPUT EVALUATOR METALLIB VISUAL_CONFIG ALPHA...}
evaluator=${2:?missing evaluator}
metallib=${3:?missing metallib}
visual=${4:?missing visual config}
shift 4
[ "$#" -gt 0 ] || {
    echo "at least one interpolation alpha is required" >&2
    exit 2
}

mkdir -p "$output/screen-low"

evaluate() {
    alpha=$1
    band=$2
    evidence="$output/screen-low/alpha-$alpha-band-$band.evidence.json"
    "$evaluator" --birdflow-american-crow-journey \
        --birdflow-journey-variant v9-visual-neural \
        --minimum-difficulty-band "$band" \
        --maximum-difficulty-band "$band" \
        --no-scheduled-resets \
        --envs 512 --steps 1600 --repeats 1 --chunk 1 \
        --seed 2650443581 \
        --metallib "$metallib" \
        --visual-observation-config "$visual" \
        --policy-pack "$output/alpha-$alpha.deployment.policypack" \
        > "$evidence" \
        2> "$output/screen-low/alpha-$alpha-band-$band.stderr.log"
}

for alpha do
    evaluate "$alpha" 2
    evidence="$output/screen-low/alpha-$alpha-band-2.evidence.json"
    if jq -e '
        .termination_count == .timeout_count
        and .mean_tracking_score >= 0.65
        and .maximum_root_height >= 0.55
        and .mean_tilt <= 0.35
        and .maximum_tilt < 0.8
    ' "$evidence" >/dev/null; then
        evaluate "$alpha" 10
        evaluate "$alpha" 3
        evaluate "$alpha" 9
    fi
done

touch "$output/screen-low/complete"
