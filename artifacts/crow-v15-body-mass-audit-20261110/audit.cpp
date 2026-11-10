#define main toe_free_admission_main
#include "../crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
#include <iomanip>
int main(){auto p=toeModel();auto&m=p.mechanics;double mass=0;std::cout<<std::setprecision(12);for(unsigned i=0;i<m.bodies.size();i++){auto x=m.bodies[i].massAndInverseMass.x;mass+=x;std::cout<<i<<" "<<m.bodyNames[i]<<" "<<x<<"\n";}std::cout<<"total "<<mass<<"\n";}
