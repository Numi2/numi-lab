# Main branch recheck

Against main base `0f3cc104f`, a fresh native build passed the expanded horizon and reset matrix with zero actions and a same-runtime zero policy. All 32 main selector unit tests passed. The full synthetic train-and-select workflow was run on the integrated revision, as documented separately; this is not a claim that the main wire format can load the integrated policy.

The first port check failed because main did not explicitly request the initial host reset in this mode. The main-specific implementation now requests it when periodic resets are disabled and scheduled resets are enabled; default main scheduling remains unchanged. The failed check is retained.

A subsequent authored-state test originally predicted that disabling host resets would retain the model default, but observed the authored x=0.9 value instead. The amended test records that value with and without host scheduling, and distinguishes 0 versus 1 requested host reset. Both amended checks and the 26 original control-flow cases pass in each execution mode. This observes a synthetic initialization contract; it does not establish the internal initialization mechanism or physical validity.
