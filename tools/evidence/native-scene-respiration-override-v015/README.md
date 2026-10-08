# Bind an assembled resting scene to its respiration parameters

The scene preflight builder now accepts a respiration configuration path and SHA-256 together. It passes the selected file to the existing native owner and binds it in the source inventory, native asset map, and post-run comparison. This is needed when a corrected lung mesh changes its source-derived diaphragm swept area. Native geometry/area admission remains unchanged.

This bundle preserves v015's original scripts, fixtures, assembly reports, comparison and revision. The default and alternate-path CPU fixtures used identical configuration bytes; neither is a changed-area mechanics test. The comparison reused the completed 20-second viewer018 run. Root verification reproduced all 17 recorded output hashes and all 27 tests passed on the SSH Mac mini.

Run the original pinned test path with the preparation Python:
```sh
/Users/n/numi-human-prep-venv-20261005/bin/python -m pytest -q -p no:cacheprovider /Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/v015/test_final_scene_936_v015.py
```

Use a fresh output path when assembling or launching. The two 29 MB composed anatomy receipts remain on the Mac mini and are listed in retained-artifacts.json. Copied source paths deliberately retain their original absolute bindings.

This change performs no new GPU run, changes no simulation equations, and does not qualify the pending lung repair, final five-minute study, or clinical physiology.
