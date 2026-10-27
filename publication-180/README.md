# P180: Correction to the P179 construction receipt

P179 captured `evidence/checkpoint-663-minimal/retention-report.json` while it was still being written; that archived file has trailing data and is not valid JSON. The immutable P179 commit and its manifest retain those exact bytes as evidence. Use `corrections/retention-report.json` here for the completed, valid receipt. The 663 candidate, its 545/130/159 checks and its strict final state were separately captured and are unaffected.

This supplement also preserves the connected-component extraction and attribution source and receipts. The result remains an unselected static construction: 663 surface pairs, zero detected skin self-pairs, zero sampled contained node–bone pairs, no native admission and no accepted physical steps.
