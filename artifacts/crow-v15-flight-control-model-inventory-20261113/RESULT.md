# Research-body flight-control inventory

The native hip-roll/two-segment research model inventory reports bilateral wing sweep, flap and pronation joints and a single tail-pitch joint. It does not contain a tail-yaw joint. It also includes bilateral hip pitch, knee and ankle pitch, hip roll, neck and two-segment toes. This inventory describes the constructed research model, not a specimen or proof that another crow model lacks those motions. Limits, drive effort and speed fields are reported directly in `all-dofs.log`.

The turn-identification experiments changed only modeled wing commands; they did not actuate a tail-yaw surface. The absence of a tail-yaw degree of freedom limits which biological steering strategies this particular body can express, but does not establish that adding one would solve the measured heading and sideslip problem. Any change to the v15 body requires new qualification and must not be silently attributed to a frozen controller.
