# Evidence retirement

Independently fetched published evidence into a separate Git checkout. Verified split or compressed hashes and decompressed bytes against each targeted local file; checked open handles and rehashed immediately before retiring only these redundant local copies. The older restored rollout used its previously independently verified manifest. The split gzip has a tracked restore script; the rollout can be restored by decompressing the tracked gzip.
