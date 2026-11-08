#include "../apps/SutureContactPublication.hpp"

static_assert(sutureChannelCountAdmissible(0, 0, 16, 16, true));
static_assert(!sutureChannelCountAdmissible(0, 0, 16, 16, false));
static_assert(sutureChannelCountAdmissible(0, 1, 16, 16, false));
static_assert(!sutureChannelCountAdmissible(0, 2, 16, 16, true));
static_assert(sutureChannelCountAdmissible(1, 1, 2, 4, true));
static_assert(sutureChannelCountAdmissible(1, 9, 2, 4, false));
static_assert(!sutureChannelCountAdmissible(1, 0, 2, 4, true));
static_assert(!sutureChannelCountAdmissible(1, 10, 2, 4, true));
int main() {}
