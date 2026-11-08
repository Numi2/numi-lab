#include "CandidatePublication.hpp"
#include <cassert>
#include <iostream>
int main(){
 CandidatePublication p(2,2);
 CandidatePublication::Certificate good{true,true,{0,0},{0,0}};
 auto c=p.begin();c.environments[0].actor={1,2};c.environments[1].actor={3,4};
 bool refusedOverlap=false;try{(void)p.begin();}catch(const std::logic_error&){refusedOverlap=true;}assert(refusedOverlap);
 auto bad=good;bad.commandCompleted=false;assert(!p.publish(c,bad));assert(p.generation()==0);
 auto stale=c;c=p.begin();assert(!p.publish(stale,good));assert(p.generation()==0);
 bad=good;bad.resultValidatedAndPublished=false;assert(!p.publish(c,bad));assert(p.generation()==0);
 c=p.begin();bad=good;bad.worldStatus={0};assert(!p.publish(c,bad));assert(p.generation()==0);
 c=p.begin();c.environments[0].actor={1,2};c.environments[1].actor={3,4};
 bad=good;bad.contactStatus={0,1};assert(p.publish(c,bad));assert(p.generation()==1);
 assert(p.records()[0].actor==std::vector<float>({1,2}) && p.records()[0].acceptedSteps==1);
 assert(p.records()[1].actor==std::vector<float>({-1234,-1234}) && p.records()[1].acceptedSteps==0);
 assert(!p.publish(c,good));assert(p.generation()==1);
 auto d=p.begin();d.environments[0].actor={7,8};d.environments[1].actor={9};
 assert(!p.publish(d,good));assert(p.generation()==1);
 d=p.begin();d.environments[0].actor={7,8};d.environments[1].actor={9,10};bad=good;bad.worldStatus={1,1};
 assert(p.publish(d,bad));assert(p.generation()==1);auto allFailed=d;
 d=p.begin();d.environments[0].actor={7,8};d.environments[1].actor={9,10};
 assert(!p.publish(allFailed,good));assert(p.publish(d,good));assert(p.generation()==2);
 assert(p.records()[0].actor[0]==7 && p.records()[1].actor[0]==9);
 std::cout<<"candidate publication unit cases passed\n";
}
