# Recovery observation regression

`observation` used an out-of-scope `duration_s` after its extraction from the
runner. This caused a post-run exception before the observation was published.
Recovery now uses the declared accepted duration (`steps * dt`). The nine
adapter tests pass on the SSH Mac mini, including distinct pre-dose, dose and
recovery plateaus that check the final window and its change from the dose.

The corrected adapter also parsed the retained `native-dt2-30s-001` trajectory
on that Mac. `retained-30s-parser-check.json` is a retrospective parser check
with arbitrary seven-second windows, not an intervention experiment. The
complete-breath metrics correctly remain unavailable in those short windows.
This check neither re-executes physics nor qualifies the known anatomy defects.

Exact command: from
`/Users/n/numi-human-resting-evidence-20261005/native-study-adapter-004`,
`/Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_resting_intervention_study`.
The remote directory retains the executed adapter and test source.
