# Authored failure termination control

The separate height-termination fixture was rejected correctly by compilation because Franka has a fixed root; retain that failure. A replacement fixed-root fixture uses a contact-group termination with threshold -1 to force a task failure, without claiming a physical collision. For one and eight steps, require reason 3 and clean rate zero, including a termination exactly at the final requested step.
