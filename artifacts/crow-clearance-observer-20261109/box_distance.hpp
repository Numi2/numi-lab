#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
namespace crowclear {
struct V { double x,y,z; };
inline V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V operator*(V a,double b){return {a.x*b,a.y*b,a.z*b};}
inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double norm2(V a){return dot(a,a);}
struct Q {double x,y,z,w;};
inline Q normalized(Q q){double n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);return {q.x/n,q.y/n,q.z/n,q.w/n};}
inline Q mul(Q a,Q b){return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z};}
inline V rotate(Q q,V v){V u{q.x,q.y,q.z};V t=cross(u,v)*2.;return v+t*q.w+cross(u,t);}
struct Box {V c; std::array<V,3> axis; std::array<double,3> h;};
inline Box box(V body,Q bodyQ,V local,Q localQ,V dimensions) {
 bodyQ=normalized(bodyQ);Q q=normalized(mul(bodyQ,normalized(localQ)));
 return {body+rotate(bodyQ,local),{rotate(q,{1,0,0}),rotate(q,{0,1,0}),rotate(q,{0,0,1})},{dimensions.x,dimensions.y,dimensions.z}};
}
inline std::array<V,8> vertices(const Box& b) {
 std::array<V,8> p{};
 for(int k=0;k<8;k++){p[k]=b.c;for(int j=0;j<3;j++)p[k]=p[k]+b.axis[j]*(b.h[j]*((k&(1<<j))?1.:-1.));}
 return p;
}
inline double pointBox2(V p,const Box& b) {
 V d=p-b.c;double total=0;
 for(int i=0;i<3;i++){double outside=std::max(0.,std::abs(dot(d,b.axis[i]))-b.h[i]);total+=outside*outside;}
 return total;
}
inline double segment2(V p,V q,V a,V b) {
 V d1=q-p,d2=b-a,r=p-a;double aa=norm2(d1),ee=norm2(d2),f=dot(d2,r),s=0,t=0;
 if(aa<=1e-24 && ee<=1e-24)return norm2(p-a);
 if(aa<=1e-24)t=std::clamp(f/ee,0.,1.);
 else {double c=dot(d1,r);if(ee<=1e-24)s=std::clamp(-c/aa,0.,1.);
 else {double bb=dot(d1,d2),den=aa*ee-bb*bb;if(den>1e-24)s=std::clamp((bb*f-c*ee)/den,0.,1.);t=(bb*s+f)/ee;
 if(t<0){t=0;s=std::clamp(-c/aa,0.,1.);} else if(t>1){t=1;s=std::clamp((bb-c)/aa,0.,1.);}}}
 return norm2((p+d1*s)-(a+d2*t));
}
inline bool separated(const Box&a,const Box&b,V axis) {
 double n2=norm2(axis);if(n2<1e-22)return false;
 double ra=0,rb=0;for(int i=0;i<3;i++){ra+=a.h[i]*std::abs(dot(axis,a.axis[i]));rb+=b.h[i]*std::abs(dot(axis,b.axis[i]));}
 return std::abs(dot(b.c-a.c,axis))>ra+rb+1e-12*std::sqrt(n2);
}
inline double distance(const Box&a,const Box&b) {
 bool apart=false;for(auto u:a.axis)apart|=separated(a,b,u);for(auto u:b.axis)apart|=separated(a,b,u);
 for(auto u:a.axis)for(auto v:b.axis)apart|=separated(a,b,cross(u,v));
 if(!apart)return 0.;
 auto av=vertices(a),bv=vertices(b);double best=std::numeric_limits<double>::infinity();
 for(V p:av)best=std::min(best,pointBox2(p,b));
 for(V p:bv)best=std::min(best,pointBox2(p,a));
 for(int ia=0;ia<8;ia++)for(int da=0;da<3;da++)if(!(ia&(1<<da)))
  for(int ib=0;ib<8;ib++)for(int db=0;db<3;db++)if(!(ib&(1<<db)))
   best=std::min(best,segment2(av[ia],av[ia|(1<<da)],bv[ib],bv[ib|(1<<db)]));
 return std::sqrt(best);
}
}
