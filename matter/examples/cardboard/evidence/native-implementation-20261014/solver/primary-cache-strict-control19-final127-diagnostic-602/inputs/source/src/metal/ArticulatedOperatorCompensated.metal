// Same constitutive and articulated operator implementation, with an explicit
// root-translation resource ABI. Legacy entry points retain their buffer ABI.
#define MR_ARTICULATED_OPERATOR_HAS_COMPENSATED_TRANSLATION 1
#define MR_ARTICULATED_OPERATOR_KERNEL_NAME mr_articulated_operator_compensated
#include "ArticulatedOperator.metal"
