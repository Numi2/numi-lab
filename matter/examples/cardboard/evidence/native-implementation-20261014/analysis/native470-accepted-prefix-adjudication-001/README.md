# Native 470 accepted-prefix adjudicator

This package is a read-only analyzer for native case 470, which is intended to
stop immediately after accepted control 19 and before control 20. It compares
the accepted prefix against retained case 469, validates the serialized
checkpoint and source167 resume metadata, and asks the source167 runner to
recompute the future regional-capture status. It does not establish checkpoint
restore parity, completion of the full coupon, or physical validation.

The analyzer deliberately does not inspect case 470 when prepared. Run it only
after the native process is terminal and the parent has explicitly confirmed
that state:

```sh
python3 analysis/native470-accepted-prefix-adjudication-001/audit.py \
  --terminal-ack '470 terminal confirmed by operator' \
  --output analysis/native470-accepted-prefix-adjudication-001/report.json
```

It refuses a live process, malformed or nonstandard JSON, modified retained
inputs, an unexpected output inventory, any nonaccepted row in controls 0–19,
or a capture presented as already observed. The report verdict is limited to
`PASS_ACCEPTED_PREFIX_ONLY`; all broader qualification flags remain false.

The exact byte comparisons cover `compiled.nmatterpack`, `initial.obj`, and
`mesh.json`. The output `manifest.json` is compared structurally after
normalizing only absolute source-root material paths; material content hashes
must still match exactly. `observations.csv` is compared byte-for-byte through
control 19, and all 40 accepted OBJ frames in that prefix must match exactly.

The package self-test is synthetic and does not read either native case:

```sh
python3 analysis/native470-accepted-prefix-adjudication-001/audit.py --self-test
```
