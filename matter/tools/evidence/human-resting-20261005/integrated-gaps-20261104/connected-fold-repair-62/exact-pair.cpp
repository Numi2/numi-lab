#include "/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanRegisteredSurfaceIntersection.hpp"
namespace S=metalrobo::human::coarse_tissue::registered_surface_intersection_detail;
extern "C" unsigned beyond(const double* points,const unsigned* fa,const unsigned* fb){
 S::Triangle a,b;S::Indices ia,ib;
 for(unsigned i=0;i<3;++i){ia[i]=fa[i];ib[i]=fb[i];for(unsigned k=0;k<3;++k){a[i][k]=points[3*ia[i]+k];b[i][k]=points[3*ib[i]+k];}}
 return S::intersectsBeyondSharedFeatures(a,ia,b,ib);
}
