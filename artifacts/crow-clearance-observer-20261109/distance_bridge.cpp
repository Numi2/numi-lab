#include "box_distance.hpp"
extern "C" double crow_distance(const double*a,const double*b) {
 using namespace crowclear;
 auto cv=[](const double*p){return box({p[0],p[1],p[2]},{p[3],p[4],p[5],p[6]},{0,0,0},{0,0,0,1},{p[7],p[8],p[9]});};
 return distance(cv(a),cv(b));
}
