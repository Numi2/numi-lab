# Split trace restoration and retirement

The three archived gzip byte parts and their manifest were fetched from an independent Git clone at the recorded remote commit. Their individual hashes and concatenated hash were verified; the concatenated gzip was independently expanded and its stream matched the original. Only the verified task-owned original gzip was removed after a fresh open-handle and hash check. The archived parts reconstruct the original via the experiment's restore-stderr.py script.
