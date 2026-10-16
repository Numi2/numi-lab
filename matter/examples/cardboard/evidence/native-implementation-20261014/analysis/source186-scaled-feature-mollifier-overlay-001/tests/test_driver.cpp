#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

struct float3 { float x,y,z; float3(float s):x(s),y(s),z(s){} float3(float X,float Y,float Z):x(X),y(Y),z(Z){} };
struct bool3 { bool x,y,z; };
static float3 operator-(float3 a,float3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static float3 operator*(float s,float3 a){return {s*a.x,s*a.y,s*a.z};}
static float3 operator/(float3 a,float s){return {a.x/s,a.y/s,a.z/s};}
static float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static float3 cross(float3 a,float3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static float max(float a,float b){return std::fmax(a,b);}
static float clamp(float x,float a,float b){return std::fmin(std::fmax(x,a),b);}
static bool3 isfinite(float3 a){return {std::isfinite(a.x),std::isfinite(a.y),std::isfinite(a.z)};}
static bool all(bool3 a){return a.x&&a.y&&a.z;}
#define thread
#include "production_mollifier_extracted.hpp"

struct D3{double x,y,z;};
static D3 sub(D3 a,D3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static D3 mul(double s,D3 a){return {s*a.x,s*a.y,s*a.z};}
static double ddot(D3 a,D3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static D3 dcross(D3 a,D3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static double norm(D3 a){return std::sqrt(ddot(a,a));}
static D3 unit(D3 a){return mul(1.0/norm(a),a);}
static void oracle(D3 a,D3 b,double& value,D3& ga,D3& gb){
 const double eta=1e-3, c=ddot(unit(a),unit(b));
 const double s2=ddot(dcross(unit(a),unit(b)),dcross(unit(a),unit(b)));
 const double r=std::clamp(s2/eta,0.0,1.0); value=r*(2-r);
 if(!(r>0&&r<1)){ga={0,0,0};gb={0,0,0};return;}
 ga=mul((-4*c*(1-r)/eta)/norm(a),sub(unit(b),mul(c,unit(a))));
 gb=mul((-4*c*(1-r)/eta)/norm(b),sub(unit(a),mul(c,unit(b))));
}
static float eval(float3 a,float3 b,float3& ga,float3& gb){return ipcFeatureMollifierDifferential(a,b,ga,gb);}
static void fail(const char* s){std::fprintf(stderr,"FAIL: %s\n",s);std::exit(1);}
int main(){
 const double theta=0.5*3.14159265358979323846/180.0;
 double maxValueErr=0, maxScaledGradErr=0, maxScaledFdErr=0;
 double values[3]{}; const double scales[3]={1e-4,1e-3,1e-2};
 for(int si=0;si<3;++si){
   const float L=(float)scales[si];
   float3 a(L,0,0), b(L*(float)std::cos(theta),L*(float)std::sin(theta),0), ga(0),gb(0);
   const float value=eval(a,b,ga,gb); values[si]=value;
   double ev;D3 ega,egb;oracle({a.x,a.y,a.z},{b.x,b.y,b.z},ev,ega,egb);
   maxValueErr=std::max(maxValueErr,std::fabs((double)value-ev));
   double ge[6]={ega.x,ega.y,ega.z,egb.x,egb.y,egb.z};
   double gv[6]={ga.x,ga.y,ga.z,gb.x,gb.y,gb.z};
      for(int k=0;k<6;++k){
      const double expectedScaled=ge[k]*L, actualScaled=gv[k]*L;
      maxScaledGradErr=std::max(maxScaledGradErr,std::fabs(expectedScaled-actualScaled));
      float3 ap[2]={a,b}, am[2]={a,b};
      const int vec=k/3, component=k%3;
      const float h=L*2.0e-4f;
      float* pp=component==0?&ap[vec].x:(component==1?&ap[vec].y:&ap[vec].z);
      float* pm=component==0?&am[vec].x:(component==1?&am[vec].y:&am[vec].z);
      *pp += h; *pm -= h;
      float3 g0(0),g1(0),g2(0),g3(0);
      const float fp=eval(ap[0],ap[1],g0,g1), fm=eval(am[0],am[1],g2,g3);
      const double fd=((double)fp-(double)fm)/(2.0*(double)h);
      maxScaledFdErr=std::max(maxScaledFdErr,std::fabs(fd*L-actualScaled));
   }
   if(!std::isfinite(value)||!std::isfinite(ga.x)||!std::isfinite(gb.y)) fail("nonfinite active result");
 }
 if(std::fabs(values[0]-values[1])>2e-6 || std::fabs(values[1]-values[2])>2e-6) fail("value is not scale invariant at 0.1x/1x/10x");
 if(maxValueErr>2e-6) fail("analytic value mismatch");
 if(maxScaledGradErr>2e-5) fail("analytic chain-rule gradient mismatch");
 if(maxScaledFdErr>2e-3) fail("finite-difference gradient mismatch");
 // Saturated orthogonal and exact degenerate-feature controls.
 float3 g1(1),g2(1);
 float saturated=eval({1e-4f,0,0},{0,1e-4f,0},g1,g2);
 if(saturated!=1.0f || dot(g1,g1)!=0 || dot(g2,g2)!=0) fail("orthogonal saturation");
 float zeroA=eval({0,0,0},{1e-3f,0,0},g1,g2);
 if(zeroA!=0.0f || dot(g1,g1)!=0 || dot(g2,g2)!=0) fail("exact zero first edge");
 float zeroB=eval({1e-3f,0,0},{0,0,0},g1,g2);
 if(zeroB!=0.0f || dot(g1,g1)!=0 || dot(g2,g2)!=0) fail("exact zero second edge");
 float parallel=eval({1e-3f,0,0},{2e-3f,0,0},g1,g2);
 if(parallel!=0.0f || dot(g1,g1)!=0 || dot(g2,g2)!=0) fail("parallel edge endpoint");
 const float nan=std::numeric_limits<float>::quiet_NaN();
 float bad=eval({nan,0,0},{1e-3f,0,0},g1,g2);
 if(std::isfinite(bad)||std::isfinite(g1.x)||std::isfinite(g2.x)) fail("nonfinite input did not fail closed");
 std::printf("PASS extracted production function: scales=0.1x,1x,10x max_value_error=%.9g max_scaled_analytic_gradient_error=%.9g max_scaled_FD_gradient_error=%.9g\n",maxValueErr,maxScaledGradErr,maxScaledFdErr);
 return 0;
}
