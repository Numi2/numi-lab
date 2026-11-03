#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using V=std::array<double,3>;
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(V a){return std::sqrt(dot(a,a));}
std::vector<double> parse(const std::string& s){std::vector<double> x;const char* p=s.c_str();char* e;while(*p){x.push_back(std::strtod(p,&e));if(e==p)throw std::runtime_error("bad CSV number");p=e;if(*p==',')++p;else if(*p)throw std::runtime_error("bad delimiter");}return x;}
int main(int argc,char** argv){try{
if(argc!=3)return 2;std::ifstream nodes(argv[1]),tets(argv[2]);if(!nodes||!tets)throw std::runtime_error("missing input");
std::string line;std::getline(nodes,line);std::vector<V> p;
while(std::getline(nodes,line)){auto x=parse(line);if(x.size()!=14||x[0]!=p.size())throw std::runtime_error("node layout");p.push_back({x[5],x[6],x[7]});}
std::getline(tets,line);std::vector<double> heights,ratios;std::array<unsigned long,4> hc{},rc{};unsigned long invalid=0,minRow=0;
double minHeight=std::numeric_limits<double>::infinity(),minVolume=minHeight;
while(std::getline(tets,line)){auto x=parse(line);if(x.size()!=9||x[0]!=heights.size())throw std::runtime_error("tet layout");
V v[4];for(int k=0;k<4;++k){auto n=std::size_t(x[3+k]);if(n>=p.size())throw std::runtime_error("node range");v[k]=p[n];}
double det=std::abs(dot(sub(v[1],v[0]),cross(sub(v[2],v[0]),sub(v[3],v[0]))));
double maxArea2=0,maxEdge=0;for(int a=0;a<4;++a)for(int b=a+1;b<4;++b){maxEdge=std::max(maxEdge,norm(sub(v[a],v[b])));for(int c=b+1;c<4;++c)maxArea2=std::max(maxArea2,norm(cross(sub(v[b],v[a]),sub(v[c],v[a]))));}
double h=det/maxArea2,r=maxEdge/h;if(!(h>0)||!std::isfinite(r))++invalid;
if(h<minHeight){minHeight=h;minRow=heights.size();}minVolume=std::min(minVolume,det/6);
heights.push_back(h);ratios.push_back(r);
for(int i=0;i<4;++i){if(h<std::pow(10.,-8+i))++hc[i];if(r>std::pow(10.,2+i))++rc[i];}
}
std::sort(heights.begin(),heights.end());std::sort(ratios.begin(),ratios.end());
std::cout<<std::setprecision(17)<<"{\"scope\":\"actual native cooked source-reference geometry; conditioning only, no acceptance waiver\",\"node_count\":"<<p.size()<<",\"tet_count\":"<<heights.size()<<",\"invalid_geometry\":"<<invalid<<",\"minimum_altitude_m\":"<<minHeight<<",\"minimum_altitude_source_row\":"<<minRow<<",\"minimum_volume_m3\":"<<minVolume<<",\"altitude_counts_below\":{\"1e-8\":"<<hc[0]<<",\"1e-7\":"<<hc[1]<<",\"1e-6\":"<<hc[2]<<",\"1e-5\":"<<hc[3]<<"},\"edge_to_height_counts_above\":{\"100\":"<<rc[0]<<",\"1000\":"<<rc[1]<<",\"10000\":"<<rc[2]<<",\"100000\":"<<rc[3]<<"},\"altitude_quantiles_m\":[";
double qs[]={0,.001,.01,.1,.5,.99,1};for(int i=0;i<7;++i){if(i)std::cout<<',';std::cout<<heights[std::size_t(qs[i]*(heights.size()-1))];}std::cout<<"]}\n";
return invalid?1:0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
