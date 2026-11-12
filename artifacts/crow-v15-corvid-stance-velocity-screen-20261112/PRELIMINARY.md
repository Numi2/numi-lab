# Analysis cutoff correction

The first analysis stopped because the newer passive host logs contacts above 1e-9 N s while the unchanged baseline comparator counts contacts above 1e-7 N s. The measurement script now filters the newer records to >1e-7 for the matched motion comparison. The feedback sensor audit separately uses all records above 1e-9. The physical run and its trace are unchanged.
