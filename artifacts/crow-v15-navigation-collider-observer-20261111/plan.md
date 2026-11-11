# Navigation collider observer

Add read-only logging of nonzero bird-to-scene contact constraints for every accepted physics substep, retaining body IDs, impulse, contact location and separation. Validate exact state and existing contact-observer parity before relying on it. A contact is not automatically a fatal collision; correlate it with termination and prior body state. Noncontacts require separate geometry-based minimum-clearance measurement. This adds no new physical force or policy input.
