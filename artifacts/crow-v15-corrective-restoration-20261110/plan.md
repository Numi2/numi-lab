# Restore and retire verified duplicate research traces

Fetch the published branch into an independent existing Git clone, bind to the expected published revision, stream each of the 15 listed compressed files from Git, decompress and hash it independently, and compare against its exact local raw counterpart. Only after all hashes agree, remove these named duplicate raw files. Preserve every Git archive, user-owned directory and all active-run files. Record the remote commit, hashes and byte counts.
