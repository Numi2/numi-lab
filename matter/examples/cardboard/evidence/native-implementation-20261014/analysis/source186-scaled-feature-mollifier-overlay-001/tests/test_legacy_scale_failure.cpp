#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
struct float3 { float x,y,z; float3():x(0),y(0),z(0){} float3(float s):x(s),y(s),z(s){} float3(float X,float Y,float Z):x(X),y(Y),z(Z){} };
static float3 operator-(float3 a,float3 b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static float3 operator*(float s,float3 a){return {s*a.x,s*a.y,s*a.z};}
static float3 operator*(float3 a,float s){return {s*a.x,s*a.y,s*a.z};}
static float3 operator/(float3 a,float s){return {a.x/s,a.y/s,a.z/s};}
static float dot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static float max(float a,float b){return std::fmax(a,b);}
static float clamp(float x,float a,float b){return std::fmin(std::fmax(x,a),b);}
constexpr float kGeometryFloor=1.0e-12f;
#define thread
#include "baseline_mollifier_extracted.hpp"
static float value(float L){float3 a(L,0,0),b(0,L,0),ga(0),gb(0);return ipcFeatureMollifierDifferential(a,b,ga,gb);}
int main(){float small=value(5.0e-4f),large=value(5.0e-3f);if(small!=0.0f||large!=1.0f){std::fprintf(stderr,"baseline mismatch: %g %g\n",small,large);return 1;}std::printf("EXPECTED BASELINE SCALE FAILURE: orthogonal .5mm=%g, 5mm=%g\n",small,large);return 0;}
