#define main toe_free_admission_main
#include "/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-toe-grip-redamping-20261110/toe_model.cpp"
#undef main
int main(){auto p=toeModel();double m=0;for(auto& b:p.mechanics.bodies) m+=b.massAndInverseMass.x;std::cout<<"mass "<<m<<"\n";}
