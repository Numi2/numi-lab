from pathlib import Path
import ast
A=Path(__file__).resolve().parent
for name in ("build.py","run.py","compare.py","posthoc-motion.py","rebind.py"):
 ast.parse((A/name).read_text(),filename=name)
assert "candidate-20261126-rebound" in (A/"compare.py").read_text()
assert "CROW_EXPERIMENT_GAIT_SMOOTH" in (A/"run.py").read_text()
print("Matched gait experiment scripts parse and use the rebound candidate.")
