# Per-column derivative gates

The finer FP64 sweep passes the full matrix norm1e-6 comparison at a common1e-10 step for both retained medium points. That does not imply every column passes: tet1634 has worst column1.90e-6 at1e-10 and1.07e-6 at1e-11. All columns and all declared steps are retained, with independent analytic matrix comparisons and explicit unresolved absolute noise floor.
