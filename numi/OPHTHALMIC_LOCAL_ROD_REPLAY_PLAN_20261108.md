# Standalone local-frame rod split-run replay

Using the same 128-node, two-offset effective axial fixture and 41 accepted updates, retain a continuous run's per-step node positions, velocities, twists, twist rates and both attachment reactions. Serialize those four state arrays after step 20, close and reopen the file, then continue from step 21 through 40 in a fresh sequence of standalone calls. Require byte-identical state and reaction arrays at every corresponding accepted continuation step. Require a malformed or truncated checkpoint to be rejected without modifying an existing destination state. Pin source, binary, owner and shader. Preserve failures.

This measures only explicit host-state serialization and continuation for the standalone contact-free fixture. It is not a MetalWorld checkpoint, a cross-process owner import, a coupled contact-cache restore, or a general save/restore guarantee; no historical surgical checkpoint is represented.
