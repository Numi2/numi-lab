// Compile the same analytic collision routines with the compact operator.
// The legacy collision library retains its separate safe math build.
#define MR_COLLISION_RESIDENT_ONLY
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "Collision.metal"
#pragma clang diagnostic pop
#include "ResidentArticulatedWorld.metalinc"
