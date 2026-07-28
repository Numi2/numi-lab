import os
import sys

from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    os.environ.setdefault(
        "METALROBO_PYTHON_EXECUTABLE",
        sys.executable,
    )
    setup(
        ext_modules=[
            extension.CMakeExtension(
                "metalrobo._mlx_ext",
                sourcedir="mlx_ext",
            )
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
        package_data={
            "metalrobo": [
                "*.so",
                "*.dylib",
                "*.metallib",
                "py.typed",
            ]
        },
        zip_safe=False,
    )
