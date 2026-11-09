#include "box_distance.hpp"
#include <cassert>
#include <cmath>
using namespace crowclear;
int main() {
 auto a=box({0,0,0},{0,0,0,1},{0,0,0},{0,0,0,1},{1,1,1});
 auto b=box({3,0,0},{0,0,0,1},{0,0,0},{0,0,0,1},{1,1,1});
 assert(std::abs(distance(a,b)-1)<1e-12);
 b=box({3,4,0},{0,0,0,1},{0,0,0},{0,0,0,1},{1,1,1});
 assert(std::abs(distance(a,b)-std::sqrt(5.))<1e-12);
 b=box({1.9,0,0},{0,0,0,1},{0,0,0},{0,0,0,1},{1,1,1});
 assert(distance(a,b)==0);
 double s=std::sqrt(.5);
 b=box({3,0,0},{0,0,s,s},{0,0,0},{0,0,0,1},{1,.5,1});
 assert(std::abs(distance(a,b)-1.5)<1e-12);
 b=box({3,0,0},{0,0,0,1},{1,0,0},{0,0,0,1},{1,1,1});
 assert(std::abs(distance(a,b)-2.)<1e-12);
 assert(std::abs(distance(a,b)-distance(b,a))<1e-12);
}
