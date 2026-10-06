#include "../apps/NumiHumanRestingCardiacIdentity.hpp"

static_assert(numiHumanRestingCardiacConceptAccepted(1,"FMA9457"));
static_assert(numiHumanRestingCardiacConceptAccepted(1,"FMA7088")); // legacy right-atrium receipt
static_assert(!numiHumanRestingCardiacConceptAccepted(1,"FMA9531"));
static_assert(numiHumanRestingCardiacConceptAccepted(24,"FMA9531"));
static_assert(numiHumanRestingCardiacConceptAccepted(24,"FMA7088")); // legacy left-atrium receipt
static_assert(!numiHumanRestingCardiacConceptAccepted(24,"FMA9457"));
static_assert(numiHumanRestingCardiacConceptAccepted(23,"FMA13884"));
static_assert(!numiHumanRestingCardiacConceptAccepted(23,"FMA7088"));
static_assert(!numiHumanRestingCardiacConceptAccepted(99,"FMA7088"));
static_assert(!numiHumanRestingCardiacConceptAccepted(1,{}));

int main() { return 0; }
