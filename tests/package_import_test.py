import os
from pathlib import Path
import subprocess
import sys
import unittest


class PackageImportTest(unittest.TestCase):
    def test_non_learning_import_does_not_initialize_mlx(self) -> None:
        python_root = Path(__file__).resolve().parents[1] / "python"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(python_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import metalrobo; "
                    "import metalrobo.ardy_onnx; "
                    "assert 'mlx' not in sys.modules; "
                    "assert metalrobo.__version__ == '0.4.0'; "
                    "assert len(dir(metalrobo)) == len(set(dir(metalrobo)))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
