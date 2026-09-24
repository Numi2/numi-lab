// Human standing reuses one angular Jacobian per body and DOF across the
// source muscle routes. The ordinary and suffix kernels retain their ABI.
#define MR_SOURCE_PAIRED_GEOMETRY 1
#define MR_MUJOCO_ANGULAR_CACHE 1
#define MR_MUJOCO_REFERENCE_KERNEL_NAME mr_mujoco_muscle_reference_cached
#include "MujocoMuscleReference.metal"
