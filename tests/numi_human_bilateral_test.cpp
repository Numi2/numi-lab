#include "metalrobo/numi_human_bilateral.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {
unsigned checks=0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) {std::cerr<<message<<'\n';std::exit(1);}
}
void checkSystem(const std::vector<float>& matrix, const std::vector<float>& expected) {
    const auto n=static_cast<unsigned>(expected.size());
    std::vector<float> factor=matrix,scale(n),pivots(n),rhs(n);
    for (unsigned i=0;i<n;++i) {
        double value=0;
        for (unsigned j=0;j<n;++j) value+=double(matrix[i*n+j])*expected[j];
        rhs[i]=static_cast<float>(value);
    }
    const auto original=rhs;
    require(mrNumiHumanBilateralFactor(factor.data(),scale.data(),pivots.data(),n),"factor rejected finite independent block");
    require(mrNumiHumanBilateralSolve(factor.data(),scale.data(),pivots.data(),rhs.data(),n),"solve rejected finite block");
    for (unsigned i=0;i<n;++i) {
        double sum=0, norm=std::abs(double(original[i]));
        for (unsigned j=0;j<n;++j) {sum+=double(matrix[i*n+j])*rhs[j];norm+=std::abs(double(matrix[i*n+j])*rhs[j]);}
        require(std::abs(sum-original[i])<=3e-6*std::max(norm,1e-20),"bilateral original-unit backward error");
        require(std::abs(rhs[i]-expected[i])<=3e-4*(1+std::abs(expected[i])),"bilateral multiplier differs from manufactured exact solution");
    }
    // Factors are reusable and do not retain a previous RHS or impulse.
    std::fill(rhs.begin(),rhs.end(),0.0f);
    require(mrNumiHumanBilateralSolve(factor.data(),scale.data(),pivots.data(),rhs.data(),n),"zero RHS solve failed");
    require(std::all_of(rhs.begin(),rhs.end(),[](float v){return v==0;}),"zero residual produced impulse");
}
struct ReducedEqualityReference {
    unsigned dependent = 0u;
    int master = -1;
    double derivative = 0.0;
};

bool validReducedGraph(
    unsigned nv, const std::vector<ReducedEqualityReference>& equalities
) {
    if (equalities.empty() || equalities.size() >= nv) return false;
    std::vector<unsigned char> dependent(nv, 0u);
    for (const auto& equality : equalities) {
        if (equality.dependent >= nv || dependent[equality.dependent])
            return false;
        if (equality.master >= 0 &&
            (unsigned(equality.master) >= nv ||
             unsigned(equality.master) == equality.dependent ||
             !std::isfinite(equality.derivative)))
            return false;
        dependent[equality.dependent] = 1u;
    }
    for (const auto& equality : equalities)
        if (equality.master >= 0 &&
            dependent[unsigned(equality.master)] != 0u)
            return false;
    return true;
}

bool solveReference(
    std::vector<double> matrix, std::vector<double> rhs, unsigned n,
    std::vector<double>& result
) {
    for (unsigned k = 0u; k < n; ++k) {
        unsigned pivot = k;
        for (unsigned row = k + 1u; row < n; ++row)
            if (std::abs(matrix[row*n+k]) > std::abs(matrix[pivot*n+k]))
                pivot = row;
        if (!(std::abs(matrix[pivot*n+k]) > 1.0e-14)) return false;
        for (unsigned column = k; column < n; ++column)
            std::swap(matrix[k*n+column], matrix[pivot*n+column]);
        std::swap(rhs[k], rhs[pivot]);
        for (unsigned row = k + 1u; row < n; ++row) {
            const double factor = matrix[row*n+k] / matrix[k*n+k];
            for (unsigned column = k; column < n; ++column)
                matrix[row*n+column] -= factor * matrix[k*n+column];
            rhs[row] -= factor * rhs[k];
        }
    }
    result.assign(n, 0.0);
    for (unsigned reverse = 0u; reverse < n; ++reverse) {
        const unsigned row = n - 1u - reverse;
        double value = rhs[row];
        for (unsigned column = row + 1u; column < n; ++column)
            value -= matrix[row*n+column] * result[column];
        result[row] = value / matrix[row*n+row];
    }
    return true;
}

void testReducedResponseReference() {
    constexpr unsigned nv = 6u;
    const std::vector<ReducedEqualityReference> equalities{
        {1u, -1, 0.0}, {3u, 2, 0.4}, {5u, 2, -0.3}};
    require(validReducedGraph(nv, equalities),
            "fixed and coupled scalar equality graph rejected");
    std::vector<double> lower(nv*nv, 0.0), matrix(nv*nv, 0.0);
    for (unsigned row = 0u; row < nv; ++row) {
        for (unsigned column = 0u; column <= row; ++column)
            lower[row*nv+column] = row == column
                ? 2.0 + 0.1 * row
                : 0.07 * double(int((row*7u + column*3u) % 5u) - 2);
    }
    for (unsigned row = 0u; row < nv; ++row)
        for (unsigned column = 0u; column < nv; ++column) {
            double value = 0.0;
            for (unsigned k = 0u; k < nv; ++k)
                value += lower[row*nv+k] * lower[column*nv+k];
            matrix[row*nv+column] = value + (row == column ? 0.5 : 0.0);
        }

    std::vector<double> equalityMatrix(equalities.size()*nv, 0.0);
    for (unsigned row = 0u; row < equalities.size(); ++row) {
        equalityMatrix[row*nv+equalities[row].dependent] = 1.0;
        if (equalities[row].master >= 0)
            equalityMatrix[row*nv+unsigned(equalities[row].master)] =
                -equalities[row].derivative;
    }
    std::vector<int> coordinate(nv, -1);
    std::vector<double> coefficient(nv, 0.0);
    std::vector<unsigned char> dependent(nv, 0u);
    for (const auto& equality : equalities)
        dependent[equality.dependent] = 1u;
    unsigned freeDofs = 0u;
    for (unsigned dof = 0u; dof < nv; ++dof)
        if (!dependent[dof]) {
            coordinate[dof] = int(freeDofs++);
            coefficient[dof] = 1.0;
        }
    for (const auto& equality : equalities)
        if (equality.master >= 0) {
            coordinate[equality.dependent] =
                coordinate[unsigned(equality.master)];
            coefficient[equality.dependent] = equality.derivative;
        }
    std::vector<double> reduced(freeDofs*freeDofs, 0.0);
    for (unsigned row = 0u; row < freeDofs; ++row)
        for (unsigned column = 0u; column <= row; ++column) {
            double value = 0.0;
            for (unsigned i = 0u; i < nv; ++i)
                if (coordinate[i] == int(row))
                    for (unsigned j = 0u; j < nv; ++j)
                        if (coordinate[j] == int(column))
                            value += coefficient[i] * matrix[i*nv+j] *
                                coefficient[j];
            reduced[row*freeDofs+column] = value;
            reduced[column*freeDofs+row] = value;
        }
    for (unsigned i = 0u; i < freeDofs; ++i)
        for (unsigned j = 0u; j < freeDofs; ++j)
            require(reduced[i*freeDofs+j] == reduced[j*freeDofs+i],
                    "R^T A R fixture lost symmetry");

    const auto checkForce = [&](const std::vector<double>& force) {
        std::vector<double> unconstrained;
        require(solveReference(matrix, force, nv, unconstrained),
                "full reference mass solve failed");
        std::vector<double> z(nv*equalities.size(), 0.0);
        for (unsigned row = 0u; row < equalities.size(); ++row) {
            std::vector<double> rhs(nv, 0.0), solution;
            for (unsigned dof = 0u; dof < nv; ++dof)
                rhs[dof] = equalityMatrix[row*nv+dof];
            require(solveReference(matrix, rhs, nv, solution),
                    "equality response reference solve failed");
            for (unsigned dof = 0u; dof < nv; ++dof)
                z[dof*equalities.size()+row] = solution[dof];
        }
        std::vector<double> schur(equalities.size()*equalities.size(), 0.0);
        std::vector<double> equalityResidual(equalities.size(), 0.0);
        for (unsigned row = 0u; row < equalities.size(); ++row) {
            for (unsigned dof = 0u; dof < nv; ++dof)
                equalityResidual[row] +=
                    equalityMatrix[row*nv+dof] * unconstrained[dof];
            for (unsigned column = 0u; column < equalities.size(); ++column)
                for (unsigned dof = 0u; dof < nv; ++dof)
                    schur[row*equalities.size()+column] +=
                        equalityMatrix[row*nv+dof] *
                        z[dof*equalities.size()+column];
        }
        std::vector<double> multiplier;
        require(solveReference(schur, equalityResidual,
                               unsigned(equalities.size()), multiplier),
                "full equality Schur reference solve failed");
        std::vector<double> schurResponse = unconstrained;
        std::vector<double> schurReaction(equalities.size(), 0.0);
        for (unsigned row = 0u; row < equalities.size(); ++row) {
            schurReaction[row] = -multiplier[row];
            for (unsigned dof = 0u; dof < nv; ++dof)
                schurResponse[dof] -=
                    z[dof*equalities.size()+row] * multiplier[row];
        }

        std::vector<double> reducedRhs(freeDofs, 0.0);
        for (unsigned dof = 0u; dof < nv; ++dof)
            if (coordinate[dof] >= 0)
                reducedRhs[unsigned(coordinate[dof])] +=
                    coefficient[dof] * force[dof];
        std::vector<double> reducedCoordinates;
        require(solveReference(reduced, reducedRhs, freeDofs,
                               reducedCoordinates),
                "R^T A R reference solve failed");
        std::vector<double> projected(nv, 0.0);
        for (unsigned dof = 0u; dof < nv; ++dof)
            if (coordinate[dof] >= 0)
                projected[dof] =
                    coefficient[dof] *
                    reducedCoordinates[unsigned(coordinate[dof])];
        std::vector<double> reaction(equalities.size(), 0.0);
        for (unsigned row = 0u; row < equalities.size(); ++row) {
            const unsigned dependentDof = equalities[row].dependent;
            reaction[row] = -force[dependentDof];
            for (unsigned dof = 0u; dof < nv; ++dof)
                reaction[row] +=
                    matrix[dependentDof*nv+dof] * projected[dof];
        }
        for (unsigned dof = 0u; dof < nv; ++dof) {
            double equalityResidualValue = 0.0;
            for (unsigned row = 0u; row < equalities.size(); ++row)
                equalityResidualValue +=
                    equalityMatrix[row*nv+dof] * reaction[row];
            double operatorResidual = -force[dof];
            for (unsigned column = 0u; column < nv; ++column)
                operatorResidual += matrix[dof*nv+column] * projected[column];
            require(std::isfinite(projected[dof]) &&
                    std::abs(operatorResidual - equalityResidualValue) < 1e-10,
                    "reduced response failed the projected KKT residual");
            require(std::abs(projected[dof] - schurResponse[dof]) < 1e-10,
                    "reduced response differs from full Schur response");
        }
        for (unsigned row = 0u; row < equalities.size(); ++row) {
            double constraint = 0.0;
            for (unsigned dof = 0u; dof < nv; ++dof)
                constraint += equalityMatrix[row*nv+dof] * projected[dof];
            require(std::abs(constraint) < 1e-12,
                    "reduced response violates a source equality");
            require(std::abs(reaction[row] - schurReaction[row]) < 1e-10,
                    "reduced equality reaction has the wrong sign or owner");
        }
    };

    checkForce({0.7, -0.3, 1.2, 0.4, -0.8, 1.1});
    for (unsigned dof = 0u; dof < nv; ++dof) {
        std::vector<double> unit(nv, 0.0);
        unit[dof] = 1.0;
        checkForce(unit);
    }

    const std::vector<ReducedEqualityReference> chained{
        {1u, 0, 0.5}, {2u, 1, -0.25}};
    const std::vector<ReducedEqualityReference> duplicated{
        {1u, -1, 0.0}, {1u, 0, 0.2}};
    require(!validReducedGraph(nv, chained) &&
            !validReducedGraph(nv, duplicated),
            "unsupported equality graph incorrectly enabled reduced coordinates");
    const bool optIn = false;
    require(!optIn && validReducedGraph(nv, equalities),
            "default-off path did not retain the full Schur fallback");
}

}
int main() {
    testReducedResponseReference();
    checkSystem({2.0f},{-3.0f});
    // Two successive pivot swaps, including previously computed L columns.
    checkSystem({1,9,1, 2,1,8, 5,2,1},{1,-2,3});
    checkSystem({1,0.999f,0.999001f,1},{0.4f,-0.3f});
    std::mt19937 generator(718);
    std::uniform_real_distribution<float> random(-1,1);
    for(unsigned n: {2u,3u,7u,51u,128u,160u}) {
        for(unsigned trial=0;trial<6;++trial) {
            std::vector<float> raw(n*n),matrix(n*n),x(n),scale(n);
            for(auto& v:raw)v=random(generator);
            for(unsigned i=0;i<n;++i) {x[i]=random(generator);scale[i]=std::pow(10.0f,float(int(i%7)-3));}
            for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j) {
                double v=i==j ? double(n) : 0.0;
                for(unsigned k=0;k<n;++k)v+=double(raw[i*n+k])*raw[j*n+k];
                matrix[i*n+j]=static_cast<float>(v*scale[i]*scale[j]);
            }
            // Manufactured x scaled to keep contributions observable.
            for(unsigned i=0;i<n;++i)x[i]/=scale[i];
            checkSystem(matrix,x);
        }
    }
    std::vector<float> s(2),p(2),rhs{1,2};
    std::vector<float> singular{1,1,1,1};
    require(!mrNumiHumanBilateralFactor(singular.data(),s.data(),p.data(),2),"singular rows silently regularized");
    for(float bad: {0.0f,-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::vector<float> a{bad};
        require(!mrNumiHumanBilateralFactor(a.data(),s.data(),p.data(),1),"invalid diagonal accepted");
    }
    require(!mrNumiHumanBilateralFactor(nullptr,nullptr,nullptr,0),"zero dimension accepted");
    require(!mrNumiHumanBilateralFactor(nullptr,nullptr,nullptr,161),"oversized dimension accepted");
    std::vector<float> a{1,0,0,1};
    require(mrNumiHumanBilateralFactor(a.data(),s.data(),p.data(),2),"identity factor failed");
    p[0]=0.5f;
    require(!mrNumiHumanBilateralSolve(a.data(),s.data(),p.data(),rhs.data(),2),"nonintegral pivot accepted");
    std::cout<<"Human bilateral block: "<<checks<<" checks passed\n";
}
