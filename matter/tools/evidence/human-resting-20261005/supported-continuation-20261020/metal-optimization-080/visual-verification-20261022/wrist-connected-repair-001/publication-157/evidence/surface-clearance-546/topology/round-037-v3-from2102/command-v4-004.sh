#!/bin/sh
set -eu
exec /Users/n/numi-human-prep-venv-20261005/bin/python /tmp/numi-human-geometry-slot.py /Users/n/numi-human-prep-venv-20261005/bin/python \
  /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/topology/round-037-v3-from2102/sources-v4/run_quality_policy025_variant_v4.py \
  --chain \
  --candidate /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/local-witness-retention-567/retained-552-001/candidate.npz \
  --audit159 /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/local-witness-retention-567/retained-552-001/full-check/audit-159/report.json \
  --audit130 /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/local-witness-retention-567/full-check/audit-130/summary.json \
  --self-pairs130 /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/local-witness-retention-567/retained-552-001/full-check/audit-130/surface-self-pairs.npy \
  --output-root /Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/topology/round-037-v3-from2102/run-004-v4 \
  --expected-candidate-sha256 b033848a0893ff45f216d150c3847e21e7db99b3a228719efbd9d31c6c8f912a \
  --expected-audit159-sha256 7d6818776a047ffc3e272f3721348fb06f8527adf1f6bba07a5439c01fc7e00e \
  --expected-audit130-sha256 251dd08c4503e8c6723ce84fa3d7507c69c2b0b2e2f61947641d140e73240591 \
  --expected-self-pairs130-sha256 55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de \
  --rounds 3 --max-candidates 2048 --max-accept 64 --prefilter-factor 1
