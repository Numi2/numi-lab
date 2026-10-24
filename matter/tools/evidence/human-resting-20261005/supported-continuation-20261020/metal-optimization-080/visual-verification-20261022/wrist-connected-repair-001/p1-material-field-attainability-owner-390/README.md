The target check rejects only mathematically guaranteed missing volumes caused by exact constant P1 cells, or a target beyond the domain volume. It is a necessary condition, not an isovalue solve or contact/physical certificate.

The pinned 014 field transfer came from 310/311's conservative LP initializer, not a solved thickness field. Its zero plateau cannot represent either requested layer volume. This guard belongs before final-field clipping, not before the constraint solver that is supposed to modify an initializer.

Run tests with PYTHONPATH=tools /Users/n/numi-human-prep-venv-20261005/bin/python tests/test_human_skin_p1_material_volume.py. The retained-field driver rechecks source hashes and prints a report without writing geometry. No simulation update path changes.
