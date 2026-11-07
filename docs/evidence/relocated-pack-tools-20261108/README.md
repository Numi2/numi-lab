# Relocated pack-tool loading

A stock relocated package from source `44c31b27` failed before parsing arguments: both WorldPack and EpisodeTwin executables exited 134 because `@rpath/libmetalrobo.dylib` had no install RPATH. The original failed logs are retained, including an initial WorldPack invocation that used an incorrect flag; the corrected `--inspect` invocation failed identically. The unmodified build-tree tool's corrected `--inspect` succeeded.

Source `2c327d2f` adds `@loader_path/../lib` to both affected install targets. After rebuilding and installing to a fresh prefix copied to another location, a CPU check verified both load paths, inspected a WorldPack, and exercised EpisodeTwin's usage path. This is not a GPU or complete application qualification.

The WorldPack and four associated typed task/actuator/sensor/reality files are a synthetic Franka pick-and-place fixture generated on CPU from the existing `task_program_check.cpp` fixture. The adapted generator and exact files/hashes are retained; it is not a G1 task and does not establish manipulation quality or physical accuracy.
