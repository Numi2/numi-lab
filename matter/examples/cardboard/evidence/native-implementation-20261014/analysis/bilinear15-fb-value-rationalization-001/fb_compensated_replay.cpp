#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "numi/matter/paper_compensated_bytecode.h"

using S = MRCompensatedScalar;
static S neg(S a) { return mrCompensatedNegate(a); }
static S sub(S a, S b) { return mrCompensatedAdd(a, neg(b)); }
static S mul(S a, S b) { return mrCompensatedMultiply(a, b); }
static S divv(S a, S b) { return mrCompensatedDivide(a, b); }
static S sqr(S a) { return mul(a, a); }
static bool less(S a, S b) {
    const S d = sub(a, b);
    return d.high < 0.0f || (d.high == 0.0f && d.low < 0.0f);
}
static float value(S a) { return a.high + a.low; }

static S yieldGap(const std::array<float,28>& p, const std::array<S,15>& z) {
    const S s11=z[6], s22=z[7], s33=z[8], s23=z[9], s13=z[10], s12=z[11];
    const S f=sqr(sub(s22,s33));
    const S g=sqr(sub(s33,s11));
    const S h=sqr(sub(s11,s22));
    S q2=mul({p[11],0.0f},f);
    q2=mrCompensatedAdd(q2,mul({p[12],0.0f},g));
    q2=mrCompensatedAdd(q2,mul({p[13],0.0f},h));
    q2=mrCompensatedAdd(q2,mul(mul({2.0f,0.0f},{p[14],0.0f}),sqr(s23)));
    q2=mrCompensatedAdd(q2,mul(mul({2.0f,0.0f},{p[15],0.0f}),sqr(s13)));
    q2=mrCompensatedAdd(q2,mul(mul({2.0f,0.0f},{p[16],0.0f}),sqr(s12)));
    q2=mrCompensatedAdd(q2,sqr({p[25],0.0f}));
    const S q=mrCompensatedSqrt(q2);
    S hard={1.0f,0.0f};
    S md=mul({p[17],0.0f},{p[19],0.0f});
    md=mul(md,{p[21],0.0f}); md=mul(md,z[13]); md=divv(md,{p[10],0.0f});
    hard=mrCompensatedAdd(hard,md);
    S cd=mul({p[18],0.0f},{p[20],0.0f});
    cd=mul(cd,{p[22],0.0f}); cd=mul(cd,z[14]); cd=divv(cd,{p[10],0.0f});
    hard=mrCompensatedAdd(hard,cd);
    return sub(hard,q);
}

int main(int argc, char** argv) {
    if(argc!=2) { std::cerr << "usage: fb_compensated_replay input.csv\n"; return 2; }
    std::ifstream in(argv[1]);
    if(!in) { std::cerr << "cannot open input\n"; return 2; }
    std::string line;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        std::istringstream stream(line);
        std::vector<float> v; std::string cell;
        while(std::getline(stream,cell,',')) v.push_back(std::stof(cell));
        if(v.size()!=60u) { std::cerr << "expected 60 fields, got " << v.size() << "\n"; return 2; }
        const unsigned iteration=static_cast<unsigned>(v[0]);
        const float native=v[1];
        std::array<float,28> p{};
        for(unsigned i=0;i<28;++i)p[i]=v[2+i];
        std::array<S,15> z{};
        for(unsigned i=0;i<15;++i)z[i]={v[30+i],v[45+i]};
        const S a=z[12];
        const S b=yieldGap(p,z);
        const S e{p[26],0.0f};
        S rad2=mrCompensatedAdd(sqr(a),sqr(b));
        rad2=mrCompensatedAdd(rad2,sqr(e));
        const S radius=mrCompensatedSqrt(rad2);
        const S directInner=sub(sub(radius,a),b);
        const S direct=mul({p[27],0.0f},directInner);
        const S sum=mrCompensatedAdd(a,b);
        const S denominator=mrCompensatedAdd(radius,sum);
        const S safeDenominator=less(denominator,e)?e:denominator;
        const S numerator=sub(sqr(e),mul({2.0f,0.0f},mul(a,b)));
        const S stable=divv(numerator,safeDenominator);
        const S selected=less(sum,{0.0f,0.0f})?direct:mul({p[27],0.0f},stable);
        std::cout << std::setprecision(17) << iteration << ',' << native << ','
                  << value(a) << ',' << value(b) << ',' << value(sum) << ','
                  << direct.high << ',' << direct.low << ',' << value(direct) << ','
                  << stable.high << ',' << stable.low << ',' << value(stable) << ','
                  << value(selected) << '\n';
    }
}
