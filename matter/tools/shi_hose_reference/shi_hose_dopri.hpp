#pragma once
#include "shi_hose_reference_generated.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace numi_shi_hose_reference {
// Independent offline FP64 Dormand–Prince5(4) integrator. Coefficients from
// Dormand & Prince1980, DOI10.1016/0771-050X(80)90013-3. It calls only the
// generated CellML RHS, never Matter kernels or native vascular equations.
struct Integrator {
    State state = initial_state();
    double time = 0.0;
    double relative_tolerance = 1e-10;
    double absolute_tolerance = 1e-10; // source coordinate units
    double maximum_step = 1e-3;
    double next_step = 1e-4;
    std::uint64_t accepted_steps = 0;
    std::uint64_t rejected_steps = 0;
    std::uint64_t evaluations = 0;
    double smallest_accepted_step = std::numeric_limits<double>::infinity();
    double largest_accepted_error_ratio = 0.0;

    void rhs(double t, const State& y, State& k) {
        evaluate(t,y,k);++evaluations;
        for(double v:k) if(!std::isfinite(v))
            throw std::runtime_error("Nonfinite generated CellML derivative");
    }
    void advance(double target) {
        if (!std::isfinite(target) || target<time || !std::isfinite(relative_tolerance) || relative_tolerance<=0 ||
            !std::isfinite(absolute_tolerance) || absolute_tolerance<=0 ||
            !std::isfinite(maximum_step) || maximum_step<=0 ||
            !std::isfinite(next_step) || next_step<=0)
            throw std::runtime_error("Invalid reference integration request");
        while(time<target) {
            const double h=std::min({next_step,maximum_step,target-time});
            if (!(h>0) || time+h==time || accepted_steps+rejected_steps>100000000ull)
                throw std::runtime_error("Source reference step underflow/budget");
            State k1{},k2{},k3{},k4{},k5{},k6{},k7{},work{},fifth{},fourth{};
            rhs(time,state,k1);
            for(std::size_t i=0;i<state_count;++i)work[i]=state[i]+h*(k1[i]/5.0);
            rhs(time+h/5.0,work,k2);
            for(std::size_t i=0;i<state_count;++i)work[i]=state[i]+h*(3.0*k1[i]/40.0+9.0*k2[i]/40.0);
            rhs(time+3.0*h/10.0,work,k3);
            for(std::size_t i=0;i<state_count;++i)work[i]=state[i]+h*(44.0*k1[i]/45.0-56.0*k2[i]/15.0+32.0*k3[i]/9.0);
            rhs(time+4.0*h/5.0,work,k4);
            for(std::size_t i=0;i<state_count;++i)work[i]=state[i]+h*(19372.0*k1[i]/6561.0-25360.0*k2[i]/2187.0+64448.0*k3[i]/6561.0-212.0*k4[i]/729.0);
            rhs(time+8.0*h/9.0,work,k5);
            for(std::size_t i=0;i<state_count;++i)work[i]=state[i]+h*(9017.0*k1[i]/3168.0-355.0*k2[i]/33.0+46732.0*k3[i]/5247.0+49.0*k4[i]/176.0-5103.0*k5[i]/18656.0);
            rhs(time+h,work,k6);
            for(std::size_t i=0;i<state_count;++i)fifth[i]=state[i]+h*(35.0*k1[i]/384.0+500.0*k3[i]/1113.0+125.0*k4[i]/192.0-2187.0*k5[i]/6784.0+11.0*k6[i]/84.0);
            rhs(time+h,fifth,k7);
            double error=0;
            for(std::size_t i=0;i<state_count;++i) {
                fourth[i]=state[i]+h*(5179.0*k1[i]/57600.0+7571.0*k3[i]/16695.0+393.0*k4[i]/640.0-92097.0*k5[i]/339200.0+187.0*k6[i]/2100.0+k7[i]/40.0);
                const double scale=absolute_tolerance+relative_tolerance*std::max(std::abs(state[i]),std::abs(fifth[i]));
                error=std::max(error,std::abs(fifth[i]-fourth[i])/scale);
                if(!std::isfinite(fifth[i])||!std::isfinite(fourth[i]))throw std::runtime_error("Nonfinite reference candidate");
            }
            if (error<=1.0) {
                state=fifth;time=(h==target-time)?target:time+h;++accepted_steps;
                smallest_accepted_step=std::min(smallest_accepted_step,h);
                largest_accepted_error_ratio=std::max(largest_accepted_error_ratio,error);
                next_step=h*(error==0?5.0:std::clamp(.9*std::pow(error,-.2),.2,5.0));
            } else {
                ++rejected_steps;next_step=h*std::clamp(.9*std::pow(error,-.2),.1,.5);
            }
        }
    }
};
}
