# Physical skin mass partition failure witnesses

The existing conservative mass transporter rejected the projected skin/fat reference mesh without identifying the material/donor/region responsible. Its rejection now retains that partition, parent/child row counts, supplied/transferred mass, child density capacities and the pinned distance bound. The transport algorithm, density limits, closure tolerances and accepted state are unchanged.

The native captured parent has 7,233 nodes, 28,632 cells and 13.939678963460464 kg. The 20 mm preparation failed for skin material 15, geometric donor 42, region 42: supplied 0.031435647722447439 kg, transferred 0.031415840393226556 kg. The total partition capacity is sufficient; local spatial capacity is insufficient. This is a failed reference preparation, not a physically admitted skin/fat volume.

Validation on SSH Numi: strict C++20/O2 builds of the existing transport check and actual child mass CLI. The existing positive fixture retains the same exact receipt hash (9e7a09107d50412732551462b9eb7dae7dcd8d39e0f5180078d13a2eb3163c96). A new disconnected-capacity case requires the correct failure partition and verifies that failure leaves the previous result unchanged. The actual 20 mm failure was reproduced with the precise witness. No GPU simulation, native motion qualification or performance gain is claimed.
