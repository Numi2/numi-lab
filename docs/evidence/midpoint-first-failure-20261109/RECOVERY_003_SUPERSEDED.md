# Recovery coordinator 003 superseded before control

Coordinator 003 was terminated while waiting for the n64 reference, before it launched any control or timing run. Review found that the existing cost runner selected the interrupted 007 control prefix while the coordinator selected the new 008 prefix. The mismatch would have prevented the timing comparison from running; it did not produce a result or alter the active n64 reference.

Coordinator 004 and cost runner 003 consistently select the 008 control. A preflight test now checks that the queued ladder, selection and timing identities agree. The physical fixtures, resource limits, accuracy gates and alternating timing schedule are unchanged. The waiting receipt and log from coordinator 003 remain preserved as evidence.
