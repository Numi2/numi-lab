#include "numi/matter/source_cylindrical_joint.h"
using namespace metal;
using namespace numi_matter_joint;
kernel void source_joint_check(device const Input<float>* inputs [[buffer(0)]],
    device const Direction<float>* directions [[buffer(1)]],
    device Output<float>* outputs [[buffer(2)]],
    device Output<float>* tangents [[buffer(3)]],
    device uint* status [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    Input<float> a=inputs[i];Direction<float> d=directions[i];
    Output<float> out{},jac{};
    status[i]=evaluate(a,d,out,jac)?1u:0u;
    outputs[i]=out;tangents[i]=jac;
}
