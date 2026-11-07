#pragma once
#include "metalrobo/gpu_types.h"
#define MR_GRU_MEMORY_ABI 1u
// Reset-after GRU, row-major gates r,z,n. Equations are fixed by this ABI.
// n=tanh(W_in*x+b_in + r*(W_hn*h+b_hn)); h'=(1-z)*n+z*h.
typedef struct MR_ALIGN16 MRGRUMemoryDispatchGPU {
    // environments, input width, hidden width, ABI.
    mr_uint4 counts;
    // read-only preview, input environment stride, reserved, reserved.
    mr_uint4 mode;
    mr_u64 inputBase, outputBase, traceBase, resetBase;
    mr_u64 parameterFingerprint, reserved;
    // absolute input admission bound, reserved.
    mr_float4 limits;
} MRGRUMemoryDispatchGPU;
// Metadata is uint4(last proposal step, episode index, initialized, reserved).
// before holds h_t; proposed holds GRU(x_t,h_t). A successful transition
// advances TaskState to t+1; only the next real inference promotes proposed.
// Preview selects the logically accepted input without writing either vector.
#ifndef __METAL_VERSION__
static_assert(sizeof(MRGRUMemoryDispatchGPU)==96u);
#endif
