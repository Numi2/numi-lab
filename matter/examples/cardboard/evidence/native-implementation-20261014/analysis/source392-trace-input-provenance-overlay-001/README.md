# Source-392 trace provenance overlay

This host-only overlay replaces the misleading fixed `source_snapshot` value with `input_coupon_source_snapshot=source-scaled-material-merit-118`. The source-392 coupon input is bound to source118; launcher/build provenance must bind the actual diagnostic build.

The exact frozen host already sets `row.input[36u]=1.0f/dt` after validating positive dt, matching the source point kernel's `rateDirection[0][0]=1/dt`. That line is retained unchanged.

The frozen source is unchanged. This artifact changes metadata only and is not a physical transaction or numerical qualification.
