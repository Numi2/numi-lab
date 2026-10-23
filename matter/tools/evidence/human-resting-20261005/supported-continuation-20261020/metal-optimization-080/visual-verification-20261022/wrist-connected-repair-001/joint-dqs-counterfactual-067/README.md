# Batched DQS counterfactual

The current294 binding scale is common to all 86 rigid bindings. Reference-exact dual-quaternion motion was evaluated on the matched native reference/shoulder/elbow captures, preserving every reference vertex. Quaternion inverse/antipodal algebra checks and native LBS replay checks pass. Script and input hashes are in the JSON.

The switch alone is insufficient: worst left shoulder compression improves from 0.37031 to 0.42350, left elbow from 0.35913 to 0.44609, but the right elbow worsens from 0.44968 to 0.42980. This has not been installed in the simulation. The next change must address source weights/geometry or physically embedded transfer rather than assuming another blend representation fixes the anatomy. Lower-joint cases without complete 86-body captures remain explicitly unmeasured.

No native images, asset edits, physics changes, contact or layer qualification. Stretch metrics alone do not establish anatomical correctness or collision freedom.

Algorithm reference: [Kavan et al.](https://users.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf), [author limitations](https://users.cs.utah.edu/~ladislav/dq/index.html).
