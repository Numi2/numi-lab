# Native respiratory intervention wiring check

The same whole-body native Metal scene was run for six simulated seconds with
normal regulation and with delivered respiratory drive multiplied by 0.5
from 0.5 to 1.5 seconds. This is an engineering connection check; the final
300-second intervention/recovery study and corrected anatomy remain required.

The seven sampled rows before dosing are byte-identical. During dosing,
diaphragm/intercostal excitation falls by half; mean diaphragm activation
falls from 0.14510 to 0.07652, intercostal activation from 0.09084 to 0.04800,
and lung air volume is 138.23 mL lower than control. Drive delivery returns
to the normal controller after 1.5 s. At 5–6 s, mean PaCO2 is 0.42137 mmHg higher,
PaO2 is 0.62875 mmHg lower, and the next breath's mean diaphragm drive is
about 2.15% higher through the existing chemoreflex. Complete gas recovery
is not demonstrated by this short run.

Both runs use the same frozen source013, circulation/respiration configuration,
and anatomy017. Source, binaries, GPU libraries, exact commands, and asset
hashes are retained in the invocation receipts. All displayed values came
from the coupled accepted state. The seven complete filling/ejection cycles
use the fixed resting cardiovascular operating point; this model does not
include respiratory venous-return changes or cardiovascular autonomic control.

The native recording retains 95 frames across 60.65 wall seconds. Its exact
Mac-mini location and hash are in retained-artifacts.json. Full accepted
traces are retained losslessly compressed. Numerical guards stayed clear;
known rib, cardiac-wall, liver, and later body-posture defects remain separate
failed anatomical gates. This is neither a completed deliverable nor clinical
validation.
