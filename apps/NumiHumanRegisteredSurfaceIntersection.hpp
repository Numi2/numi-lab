#pragma once
// Exact-sign predicates for admission of registered source surfaces. This is
// initialization validation, not an alternate physical update. Expansions retain
// binary64 subtraction/product residuals; no anatomical distance tolerance turns
// a near-coplanar pair into an intersection. Common indexed features are allowed
// only when every intersection is confined to that feature.
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>
namespace metalrobo::human::coarse_tissue::registered_surface_intersection_detail {
using Point=std::array<double,3>;
using Triangle=std::array<Point,3>;
using Indices=std::array<std::uint32_t,3>;
using Expansion=std::vector<double>;
inline void grow(Expansion& values,double q){
    Expansion result;result.reserve(values.size()+1u);
    for(const double e:values){
        const double sum=q+e,b=sum-q;
        const double error=(q-(sum-b))+(e-b);
        if(error!=0.0)result.push_back(error);
        q=sum;
    }
    if(q!=0.0||result.empty())result.push_back(q);
    values=std::move(result);
}
inline Expansion difference(double a,double b){Expansion out{a};grow(out,-b);return out;}
inline Expansion product(const Expansion& a,const Expansion& b){
    Expansion out;
    for(double x:a)for(double y:b){const double hi=x*y,lo=std::fma(x,y,-hi);grow(out,lo);grow(out,hi);}
    return out;
}
inline void accumulate(Expansion& out,const Expansion& term,double sign=1.0){for(double x:term)grow(out,sign*x);}
inline int sign(const Expansion& value){for(auto i=value.rbegin();i!=value.rend();++i)if(*i!=0.0)return *i>0.0?1:-1;return 0;}
inline int orient2(const Point& a,const Point& b,const Point& c,unsigned x,unsigned y){
    auto det=product(difference(b[x],a[x]),difference(c[y],a[y]));
    accumulate(det,product(difference(b[y],a[y]),difference(c[x],a[x])),-1.0);return sign(det);
}
inline int orient3(const Point& a,const Point& b,const Point& c,const Point& d){
    const std::array<Expansion,3> u{difference(b[0],a[0]),difference(b[1],a[1]),difference(b[2],a[2])};
    const std::array<Expansion,3> v{difference(c[0],a[0]),difference(c[1],a[1]),difference(c[2],a[2])};
    const std::array<Expansion,3> w{difference(d[0],a[0]),difference(d[1],a[1]),difference(d[2],a[2])};
    Expansion det;
    for(unsigned axis=0;axis<3u;++axis){const auto j=(axis+1u)%3u,k=(axis+2u)%3u;
        auto minor=product(v[j],w[k]);accumulate(minor,product(v[k],w[j]),-1.0);accumulate(det,product(u[axis],minor));}
    return sign(det);
}
inline bool sameStrictSide(const std::array<int,3>& s){return (s[0]>0&&s[1]>0&&s[2]>0)||(s[0]<0&&s[1]<0&&s[2]<0);}
inline bool allZero(const std::array<int,3>& s){return s[0]==0&&s[1]==0&&s[2]==0;}
inline std::array<unsigned,2> projection(const Triangle& t){
    const Point a{t[1][0]-t[0][0],t[1][1]-t[0][1],t[1][2]-t[0][2]},b{t[2][0]-t[0][0],t[2][1]-t[0][1],t[2][2]-t[0][2]};
    const Point n{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    const unsigned drop=std::abs(n[0])>=std::abs(n[1])&&std::abs(n[0])>=std::abs(n[2])?0u:(std::abs(n[1])>=std::abs(n[2])?1u:2u);
    return drop==0u?std::array<unsigned,2>{1u,2u}:(drop==1u?std::array<unsigned,2>{0u,2u}:std::array<unsigned,2>{0u,1u});
}
inline bool inside(const Point& p,const Triangle& t,const std::array<unsigned,2>& axes){
    const std::array<int,3> s{orient2(t[0],t[1],p,axes[0],axes[1]),orient2(t[1],t[2],p,axes[0],axes[1]),orient2(t[2],t[0],p,axes[0],axes[1])};
    return (s[0]>=0&&s[1]>=0&&s[2]>=0)||(s[0]<=0&&s[1]<=0&&s[2]<=0);
}
inline bool commonPoint(const Point& p,const Triangle& a,const Indices& ia,const Triangle& b,const Indices& ib){
    for(unsigned i=0;i<3u;++i)for(unsigned j=0;j<3u;++j)if(ia[i]==ib[j]&&a[i]==b[j]&&p==a[i])return true;
    return false;
}
inline bool coplanarIntersection(const Triangle& a,const Indices& ia,const Triangle& b,const Indices& ib){
    const auto axes=projection(a);
    for(unsigned i=0;i<3u;++i){
        if(inside(a[i],b,axes)&&!commonPoint(a[i],a,ia,b,ib))return true;
        if(inside(b[i],a,axes)&&!commonPoint(b[i],a,ia,b,ib))return true;
    }
    for(unsigned i=0;i<3u;++i)for(unsigned j=0;j<3u;++j){
        const auto in=(i+1u)%3u,jn=(j+1u)%3u;
        const int u=orient2(a[i],a[in],b[j],axes[0],axes[1]),v=orient2(a[i],a[in],b[jn],axes[0],axes[1]);
        const int w=orient2(b[j],b[jn],a[i],axes[0],axes[1]),z=orient2(b[j],b[jn],a[in],axes[0],axes[1]);
        if(u*v<0&&w*z<0)return true;
    }
    return false;
}
inline bool edgeIntersection(const Point& p,const Point& q,int dp,int dq,const Triangle& t,const Triangle& a,const Indices& ia,const Triangle& b,const Indices& ib){
    if(dp*dq>0)return false;
    const auto axes=projection(t);
    if(dp==0&&inside(p,t,axes)&&!commonPoint(p,a,ia,b,ib))return true;
    if(dq==0&&inside(q,t,axes)&&!commonPoint(q,a,ia,b,ib))return true;
    if(dp*dq<0){const std::array<int,3> s{orient3(p,q,t[0],t[1]),orient3(p,q,t[1],t[2]),orient3(p,q,t[2],t[0])};
        return (s[0]>=0&&s[1]>=0&&s[2]>=0)||(s[0]<=0&&s[1]<=0&&s[2]<=0);}
    if(dp==0&&dq==0)for(unsigned i=0;i<3u;++i){const auto j=(i+1u)%3u;
        if(orient2(p,q,t[i],axes[0],axes[1])*orient2(p,q,t[j],axes[0],axes[1])<0&&
           orient2(t[i],t[j],p,axes[0],axes[1])*orient2(t[i],t[j],q,axes[0],axes[1])<0)return true;}
    return false;
}
inline bool intersectsBeyondSharedFeatures(const Triangle& a,const Indices& ia,const Triangle& b,const Indices& ib){
    const std::array<int,3> sa{orient3(a[0],a[1],a[2],b[0]),orient3(a[0],a[1],a[2],b[1]),orient3(a[0],a[1],a[2],b[2])};
    if(sameStrictSide(sa))return false;
    const std::array<int,3> sb{orient3(b[0],b[1],b[2],a[0]),orient3(b[0],b[1],b[2],a[1]),orient3(b[0],b[1],b[2],a[2])};
    if(sameStrictSide(sb))return false;
    if(allZero(sa)&&allZero(sb))return coplanarIntersection(a,ia,b,ib);
    for(unsigned i=0;i<3u;++i){const auto j=(i+1u)%3u;
        if(edgeIntersection(a[i],a[j],sb[i],sb[j],b,a,ia,b,ib)||edgeIntersection(b[i],b[j],sa[i],sa[j],a,a,ia,b,ib))return true;}
    return false;
}
} // namespace metalrobo::human::coarse_tissue::registered_surface_intersection_detail
