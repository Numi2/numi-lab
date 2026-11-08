import json
import mujoco
import numpy as np

xml = """<mujoco><option timestep="0.001" gravity="0 -9.81 0" integrator="Euler"/>
<worldbody><body pos="0 10 0"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/></body></worldbody></mujoco>"""
model = mujoco.MjModel.from_xml_string(xml)
model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
data = mujoco.MjData(model)
for _ in range(1000):
    mujoco.mj_step(model, data)
expected_y = 10 - 9.81 * .001 * .001 * 1000 * 1001 / 2
expected_v = -9.81
y, v = float(data.qpos[1]), float(data.qvel[1])
result = {"engine": "mujoco", "version": mujoco.__version__,
          "numpy": np.__version__, "steps": 1000,
          "y_m": y, "vy_m_s": v,
          "position_error_m": abs(y - expected_y), "velocity_error_m_s": abs(v - expected_v),
          "passed": abs(y-expected_y) <= .001 and abs(v-expected_v) <= .001}
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
