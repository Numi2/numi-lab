#include "NumiHumanRestingSourceAllocation.hpp"
#include <cassert>
#include <iostream>

template<class F> void rejected(F function) {
    bool failed=false;try { function(); } catch(const std::runtime_error&) { failed=true; }
    assert(failed);
}
int main() {
    const std::map<unsigned,unsigned> channels{{0,12},{1,12},{4,13}};
    const std::set<unsigned> referenced{101,102,104};
    const auto result=numiHumanRestingReferencedSourceAttachmentVertices(channels,referenced,100,5);
    assert((result==std::set<unsigned>{1,4}));
    assert(channels.size()==3&&channels.contains(0));
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices(channels,{101,102},100,5);});
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices(channels,{99,101,104},100,5);});
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices(channels,{101,104,105},100,5);});
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices({{5,12}},referenced,100,5);});
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices({{1,24}},referenced,100,5);});
    rejected([&]{numiHumanRestingReferencedSourceAttachmentVertices(channels,{},100,5);});
    std::cout<<"PASS retained unused attachment slots and six invalid reference/channel cases\n";
}
