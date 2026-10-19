# Registered skin/fat thickness fields

The existing skin/fat preparation owner now accepts caller-supplied per-source-vertex thickness fields, provenance and explicit min/max summaries. It builds the same shared-interface FEM object and preserves source ancestry, contact-node eligibility and volume-based mass debits. The scalar entrypoint delegates to a constant field, whose generated geometry and bookkeeping match bitwise in the regression.

The strict CPU regression covers variable fields, rotated donor frames, shared materials/nodes, mass closure and rejection of malformed, nonfinite, out-of-range, inverted and singular inputs. Exact source/test pins and compile arguments are in cpu-test-receipt.json. No runtime library is linked by this CPU fixture.

This is preparation support for whole-body regional thickness, not a measured-person thickness map or whole-body runtime/contact qualification. No new anatomical thickness measurements are claimed, and production selection remains the existing regional patch until the full source/registration repair is integrated. No visual verification was performed.
