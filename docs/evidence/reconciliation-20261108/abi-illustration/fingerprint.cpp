#include "metalrobo/RuntimeAbi.hpp"
#include <cstdio>
int main() { std::printf("%016llx\n", static_cast<unsigned long long>(metalrobo::runtimeAbiFingerprint())); }
