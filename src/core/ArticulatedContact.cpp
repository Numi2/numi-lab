#include "metalrobo/ArticulatedContact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ContactFrame {
    Vec3 normal;
    Vec3 tangentU;
    Vec3 tangentV;
};

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

double dot(const Vec3 left, const Vec3 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

template <std::size_t Size>
bool finite(const std::array<double, Size>& values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

bool finite(const std::span<const double> values) {
    return std::ranges::all_of(values, [](const double value) {
        return finite(value);
    });
}

ArticulatedContactDiagnostics diagnosticsFor(
    const std::uint32_t articulationIndex,
    const std::size_t contactCount,
    const std::size_t nv
) {
    ArticulatedContactDiagnostics result;
    result.articulationIndex = articulationIndex;
    result.contactCount =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            contactCount,
            std::numeric_limits<std::uint32_t>::max()
        ));
    result.nv = static_cast<std::uint32_t>(std::min<std::size_t>(
        nv,
        std::numeric_limits<std::uint32_t>::max()
    ));
    return result;
}

bool makeFrame(
    const ArticulatedContact& contact,
    ContactFrame& frame
) {
    const Vec3 inputNormal = vector(contact.normal);
    const Vec3 inputTangentU = vector(contact.tangentU);
    const Vec3 inputTangentV = vector(contact.tangentV);
    if (!finite(inputNormal) ||
        !finite(inputTangentU) ||
        !finite(inputTangentV)) {
        return false;
    }
    const double normalLength = norm(inputNormal);
    const double tangentULength = norm(inputTangentU);
    const double tangentVLength = norm(inputTangentV);
    constexpr double directionTolerance = 2.0e-4;
    constexpr double orthogonalityTolerance = 4.0e-4;
    constexpr double handednessTolerance = 6.0e-4;
    if (!finite(normalLength) ||
        !finite(tangentULength) ||
        !finite(tangentVLength) ||
        std::abs(normalLength - 1.0) > directionTolerance ||
        std::abs(tangentULength - 1.0) > directionTolerance ||
        std::abs(tangentVLength - 1.0) > directionTolerance ||
        std::abs(dot(inputNormal, inputTangentU)) >
            orthogonalityTolerance ||
        std::abs(dot(inputNormal, inputTangentV)) >
            orthogonalityTolerance ||
        std::abs(dot(inputTangentU, inputTangentV)) >
            orthogonalityTolerance ||
        std::abs(
            dot(cross(inputNormal, inputTangentU), inputTangentV) -
                1.0
        ) > handednessTolerance) {
        return false;
    }
    frame.normal = inputNormal;
    frame.tangentU = inputTangentU;
    frame.tangentV = inputTangentV;
    return true;
}

struct CholeskyFactor {
    std::vector<double> lower;
    std::size_t dimension = 0u;
    double minimumPivot = 0.0;
    double maximumPivot = 0.0;
    bool valid = false;
};

CholeskyFactor factorize(
    const std::span<const double> matrix,
    const std::size_t dimension
) {
    CholeskyFactor result;
    result.dimension = dimension;
    if (dimension == 0u ||
        matrix.size() != dimension * dimension ||
        !finite(matrix)) {
        return result;
    }
    double matrixScale = 0.0;
    for (const double value : matrix) {
        matrixScale = std::max(matrixScale, std::abs(value));
    }
    if (!(matrixScale > 0.0) || !finite(matrixScale)) {
        return result;
    }
    result.lower.assign(dimension * dimension, 0.0);
    result.minimumPivot = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u; column <= row; ++column) {
            double value = matrix[row * dimension + column];
            for (std::size_t inner = 0u;
                 inner < column;
                 ++inner) {
                value -=
                    result.lower[row * dimension + inner] *
                    result.lower[column * dimension + inner];
            }
            if (row == column) {
                const double threshold =
                    64.0 * std::numeric_limits<double>::epsilon() *
                    matrixScale *
                    static_cast<double>(dimension);
                if (!(value > threshold) || !finite(value)) {
                    return result;
                }
                const double pivot = std::sqrt(value);
                result.lower[row * dimension + column] = pivot;
                result.minimumPivot =
                    std::min(result.minimumPivot, pivot);
                result.maximumPivot =
                    std::max(result.maximumPivot, pivot);
            } else {
                result.lower[row * dimension + column] =
                    value /
                    result.lower[column * dimension + column];
            }
        }
    }
    result.valid =
        finite(result.lower) &&
        finite(result.minimumPivot) &&
        finite(result.maximumPivot);
    return result;
}

bool solve(
    const CholeskyFactor& factor,
    const std::span<const double> right,
    std::vector<double>& solution
) {
    if (!factor.valid || right.size() != factor.dimension ||
        !finite(right)) {
        return false;
    }
    const std::size_t dimension = factor.dimension;
    std::vector<double> intermediate(dimension, 0.0);
    for (std::size_t row = 0u; row < dimension; ++row) {
        double value = right[row];
        for (std::size_t column = 0u;
             column < row;
             ++column) {
            value -=
                factor.lower[row * dimension + column] *
                intermediate[column];
        }
        intermediate[row] =
            value / factor.lower[row * dimension + row];
    }
    solution.assign(dimension, 0.0);
    for (std::size_t reverse = 0u;
         reverse < dimension;
         ++reverse) {
        const std::size_t row = dimension - 1u - reverse;
        double value = intermediate[row];
        for (std::size_t column = row + 1u;
             column < dimension;
             ++column) {
            value -=
                factor.lower[column * dimension + row] *
                solution[column];
        }
        solution[row] =
            value / factor.lower[row * dimension + row];
    }
    return finite(solution);
}

bool operatorStructurallyValid(
    const ArticulatedContactProblem& problem
) {
    const std::size_t nv = problem.nv;
    const std::size_t contacts = problem.contactCount;
    const std::size_t rows = 3u * contacts;
    if (nv == 0u ||
        contacts == 0u ||
        problem.massCholeskyLower.size() != nv * nv ||
        problem.contactJacobian.size() != rows * nv ||
        problem.delassus.size() != rows * rows ||
        problem.pointA.size() != contacts ||
        problem.pointB.size() != contacts ||
        !finite(problem.massCholeskyLower) ||
        !finite(problem.contactJacobian) ||
        !finite(problem.delassus)) {
        return false;
    }
    for (std::size_t row = 0u; row < nv; ++row) {
        if (!(problem.massCholeskyLower[row * nv + row] > 0.0)) {
            return false;
        }
        for (std::size_t column = row + 1u;
             column < nv;
             ++column) {
            if (problem.massCholeskyLower[row * nv + column] != 0.0) {
                return false;
            }
        }
    }
    return true;
}

CholeskyFactor retainedFactor(
    const ArticulatedContactProblem& problem
) {
    CholeskyFactor factor;
    factor.dimension = problem.nv;
    factor.lower = problem.massCholeskyLower;
    factor.minimumPivot = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0u; row < factor.dimension; ++row) {
        const double pivot =
            factor.lower[row * factor.dimension + row];
        factor.minimumPivot = std::min(
            factor.minimumPivot,
            pivot
        );
        factor.maximumPivot = std::max(
            factor.maximumPivot,
            pivot
        );
    }
    factor.valid =
        factor.dimension > 0u &&
        factor.lower.size() ==
            factor.dimension * factor.dimension &&
        finite(factor.lower) &&
        factor.minimumPivot > 0.0 &&
        finite(factor.minimumPivot) &&
        finite(factor.maximumPivot);
    return factor;
}

} // namespace

ArticulatedContactDiagnostics buildArticulatedContactProblem(
    const EngineModel& model,
    const std::uint32_t articulationIndex,
    const std::span<const double> q,
    const std::span<const double> freeVelocity,
    const std::span<const ArticulatedContact> contacts,
    ArticulatedContactProblem& problem,
    const ArticulatedDynamicsConfig& config,
    const bool buildDenseInverseCompatibilityAdapter
) {
    const std::size_t nv =
        articulationIndex < model.articulations.size()
        ? model.articulations[articulationIndex].nv
        : 0u;
    ArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            articulationIndex,
            contacts.size(),
            nv
        );
    if (contacts.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        freeVelocity.size() != nv) {
        diagnostics.status =
            ArticulatedContactStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(freeVelocity)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }
    if (articulationIndex >= model.articulations.size()) {
        diagnostics.status =
            ArticulatedContactStatus::dynamicsFailure;
        diagnostics.dynamicsStatus =
            ArticulatedDynamicsStatus::invalidModel;
        return diagnostics;
    }
    const MRArticulationGPU& articulation =
        model.articulations[articulationIndex];
    const auto ownedBody = [&articulation](
        const std::uint32_t body
    ) {
        return body >= articulation.firstBody &&
            body < articulation.firstBody + articulation.bodyCount;
    };

    std::vector<ContactFrame> frames(contacts.size());
    std::vector<ArticulatedPointQuery> queries;
    queries.reserve(2u * contacts.size());
    std::vector<std::size_t> queryA(contacts.size());
    std::vector<std::size_t> queryB(
        contacts.size(),
        std::numeric_limits<std::size_t>::max()
    );
    for (std::size_t contactIndex = 0u;
         contactIndex < contacts.size();
         ++contactIndex) {
        const ArticulatedContact& contact = contacts[contactIndex];
        if (!ownedBody(contact.bodyA) ||
            (contact.bodyB != kArticulatedStaticWorld &&
             !ownedBody(contact.bodyB)) ||
            (contact.bodyB != kArticulatedStaticWorld &&
             contact.bodyA == contact.bodyB) ||
            !finite(contact.localPointA) ||
            !finite(contact.localPointB) ||
            !finite(contact.targetVelocity) ||
            !finite(contact.regularization) ||
            !finite(contact.warmImpulse) ||
            !finite(contact.friction) ||
            !(contact.friction >= 0.0) ||
            !std::ranges::all_of(
                contact.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            ) ||
            !makeFrame(contact, frames[contactIndex])) {
            diagnostics.status =
                ArticulatedContactStatus::invalidContact;
            return diagnostics;
        }
        queryA[contactIndex] = queries.size();
        queries.push_back({
            contact.bodyA,
            contact.localPointA,
        });
        if (contact.bodyB != kArticulatedStaticWorld) {
            queryB[contactIndex] = queries.size();
            queries.push_back({
                contact.bodyB,
                contact.localPointB,
            });
        }
    }

    std::vector<ArticulatedPointKinematics> queryKinematics(
        queries.size()
    );
    std::vector<double> queryJacobians(
        queries.size() * 3u * nv,
        0.0
    );
    if (!queries.empty()) {
        const ArticulatedDynamicsDiagnostics
            kinematicsDiagnostics =
                computeArticulatedPointJacobians(
                    model,
                    articulationIndex,
                    q,
                    freeVelocity,
                    queries,
                    queryKinematics,
                    queryJacobians,
                    config
                );
        if (!kinematicsDiagnostics.succeeded()) {
            diagnostics.status =
                ArticulatedContactStatus::dynamicsFailure;
            diagnostics.dynamicsStatus =
                kinematicsDiagnostics.status;
            return diagnostics;
        }
    }

    std::vector<double> massMatrix(nv * nv, 0.0);
    const ArticulatedDynamicsDiagnostics massDiagnostics =
        computeArticulatedMassMatrix(
            model,
            articulationIndex,
            q,
            massMatrix,
            config
        );
    if (!massDiagnostics.succeeded()) {
        diagnostics.status =
            ArticulatedContactStatus::dynamicsFailure;
        diagnostics.dynamicsStatus = massDiagnostics.status;
        return diagnostics;
    }
    const CholeskyFactor factor = factorize(massMatrix, nv);
    if (!factor.valid) {
        diagnostics.status =
            ArticulatedContactStatus::factorizationFailure;
        return diagnostics;
    }
    diagnostics.minimumCholeskyPivot = factor.minimumPivot;
    diagnostics.maximumCholeskyPivot = factor.maximumPivot;

    std::vector<double> right(nv, 0.0);
    std::vector<double> column;
    std::vector<double> inverseMass;
    if (buildDenseInverseCompatibilityAdapter) {
        inverseMass.assign(nv * nv, 0.0);
        for (std::size_t sourceColumn = 0u;
             sourceColumn < nv;
             ++sourceColumn) {
            std::ranges::fill(right, 0.0);
            right[sourceColumn] = 1.0;
            if (!solve(factor, right, column)) {
                diagnostics.status =
                    ArticulatedContactStatus::factorizationFailure;
                return diagnostics;
            }
            for (std::size_t row = 0u; row < nv; ++row) {
                inverseMass[row * nv + sourceColumn] = column[row];
            }
        }
        for (std::size_t row = 0u; row < nv; ++row) {
            for (std::size_t columnIndex = row + 1u;
                 columnIndex < nv;
                 ++columnIndex) {
                const double symmetric = 0.5 * (
                    inverseMass[row * nv + columnIndex] +
                    inverseMass[columnIndex * nv + row]
                );
                inverseMass[row * nv + columnIndex] = symmetric;
                inverseMass[columnIndex * nv + row] = symmetric;
            }
        }
        for (std::size_t row = 0u; row < nv; ++row) {
            for (std::size_t columnIndex = 0u;
                 columnIndex < nv;
                 ++columnIndex) {
                double value = 0.0;
                for (std::size_t inner = 0u; inner < nv; ++inner) {
                    value +=
                        massMatrix[row * nv + inner] *
                        inverseMass[inner * nv + columnIndex];
                }
                diagnostics.maximumDenseInverseAdapterResidual = std::max(
                    diagnostics.maximumDenseInverseAdapterResidual,
                    std::abs(
                        value -
                        (row == columnIndex ? 1.0 : 0.0)
                    )
                );
            }
        }
        if (!finite(
                diagnostics.maximumDenseInverseAdapterResidual
            ) ||
            diagnostics.maximumDenseInverseAdapterResidual > 1.0e-9) {
            diagnostics.status =
                ArticulatedContactStatus::factorizationFailure;
            return diagnostics;
        }
    }

    const std::size_t rowCount = 3u * contacts.size();
    std::vector<double> jacobian(rowCount * nv, 0.0);
    std::vector<ArticulatedPointKinematics> pointA(
        contacts.size()
    );
    std::vector<ArticulatedPointKinematics> pointB(
        contacts.size()
    );
    DenseConicProblem conic;
    conic.nv = static_cast<std::uint32_t>(nv);
    conic.inverseMass = std::move(inverseMass);
    conic.freeVelocity.assign(
        freeVelocity.begin(),
        freeVelocity.end()
    );
    conic.contacts.resize(contacts.size());

    for (std::size_t contactIndex = 0u;
         contactIndex < contacts.size();
         ++contactIndex) {
        const ArticulatedContact& contact = contacts[contactIndex];
        pointA[contactIndex] = queryKinematics[queryA[contactIndex]];
        if (queryB[contactIndex] !=
            std::numeric_limits<std::size_t>::max()) {
            pointB[contactIndex] =
                queryKinematics[queryB[contactIndex]];
        } else {
            pointB[contactIndex].position = contact.localPointB;
            pointB[contactIndex].linearVelocity = {0.0, 0.0, 0.0};
        }

        const std::array<Vec3, 3> axes{
            frames[contactIndex].normal,
            frames[contactIndex].tangentU,
            frames[contactIndex].tangentV,
        };
        DenseContactBlock& block = conic.contacts[contactIndex];
        block.normalJacobian.assign(nv, 0.0);
        block.tangentUJacobian.assign(nv, 0.0);
        block.tangentVJacobian.assign(nv, 0.0);
        const std::array<std::span<double>, 3> blockRows{
            block.normalJacobian,
            block.tangentUJacobian,
            block.tangentVJacobian,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                const std::size_t aBase =
                    queryA[contactIndex] * 3u * nv;
                Vec3 relativeColumn{
                    -queryJacobians[aBase + 0u * nv + dof],
                    -queryJacobians[aBase + 1u * nv + dof],
                    -queryJacobians[aBase + 2u * nv + dof],
                };
                if (queryB[contactIndex] !=
                    std::numeric_limits<std::size_t>::max()) {
                    const std::size_t bBase =
                        queryB[contactIndex] * 3u * nv;
                    relativeColumn.x +=
                        queryJacobians[bBase + 0u * nv + dof];
                    relativeColumn.y +=
                        queryJacobians[bBase + 1u * nv + dof];
                    relativeColumn.z +=
                        queryJacobians[bBase + 2u * nv + dof];
                }
                const double value =
                    dot(axes[axis], relativeColumn);
                const std::size_t row =
                    3u * contactIndex + axis;
                jacobian[row * nv + dof] = value;
                blockRows[axis][dof] = value;
            }
        }
        block.targetVelocity = contact.targetVelocity;
        block.regularization = contact.regularization;
        block.warmImpulse = contact.warmImpulse;
        block.friction = contact.friction;
    }

    // Retain the solve response for every J' basis column. This constructs W
    // from the factorized operator itself; DenseConicProblem's inverseMass is
    // only a compatibility adapter and is not used here.
    std::vector<double> inverseMassJacobianTranspose(
        rowCount * nv,
        0.0
    );
    for (std::size_t contactRow = 0u;
         contactRow < rowCount;
         ++contactRow) {
        std::ranges::copy(
            std::span(
                jacobian.data() + contactRow * nv,
                nv
            ),
            right.begin()
        );
        if (!solve(factor, right, column)) {
            diagnostics.status =
                ArticulatedContactStatus::factorizationFailure;
            return diagnostics;
        }
        std::ranges::copy(
            column,
            inverseMassJacobianTranspose.begin() +
                contactRow * nv
        );
    }
    std::vector<double> delassus(
        rowCount * rowCount,
        0.0
    );
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t columnIndex = 0u;
             columnIndex < rowCount;
             ++columnIndex) {
            double value = 0.0;
            for (std::size_t inner = 0u;
                 inner < nv;
                 ++inner) {
                value +=
                    jacobian[row * nv + inner] *
                    inverseMassJacobianTranspose[
                        columnIndex * nv + inner
                    ];
            }
            delassus[row * rowCount + columnIndex] = value;
        }
    }
    double delassusScale = 1.0;
    for (const double value : delassus) {
        delassusScale = std::max(
            delassusScale,
            std::abs(value)
        );
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t columnIndex = row + 1u;
             columnIndex < rowCount;
             ++columnIndex) {
            diagnostics.maximumDelassusAsymmetry = std::max(
                diagnostics.maximumDelassusAsymmetry,
                std::abs(
                    delassus[row * rowCount + columnIndex] -
                    delassus[columnIndex * rowCount + row]
                )
            );
            const double symmetric = 0.5 * (
                delassus[row * rowCount + columnIndex] +
                delassus[columnIndex * rowCount + row]
            );
            delassus[row * rowCount + columnIndex] = symmetric;
            delassus[columnIndex * rowCount + row] = symmetric;
        }
    }
    const double asymmetryTolerance =
        4096.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(nv, rowCount)) *
        delassusScale;
    if (!finite(diagnostics.maximumDelassusAsymmetry) ||
        diagnostics.maximumDelassusAsymmetry >
            asymmetryTolerance) {
        diagnostics.status =
            ArticulatedContactStatus::factorizationFailure;
        return diagnostics;
    }
    if (!finite(conic.inverseMass) || !finite(jacobian) ||
        !finite(delassus)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteResult;
        return diagnostics;
    }

    ArticulatedContactProblem result;
    result.articulationIndex = articulationIndex;
    result.nv = static_cast<std::uint32_t>(nv);
    result.contactCount =
        static_cast<std::uint32_t>(contacts.size());
    result.conic = std::move(conic);
    result.massCholeskyLower = factor.lower;
    result.contactJacobian = std::move(jacobian);
    result.delassus = std::move(delassus);
    result.pointA = std::move(pointA);
    result.pointB = std::move(pointB);
    diagnostics.materializedDenseInverse =
        buildDenseInverseCompatibilityAdapter;
    problem = std::move(result);
    return diagnostics;
}

ArticulatedContactDiagnostics buildArticulatedContactSpaceProblem(
    const ArticulatedContactProblem& articulated,
    const std::span<const double> freeVelocity,
    ContactSpaceConicProblem& contactSpace
) {
    const std::size_t contactCount = articulated.contactCount;
    const std::size_t rowCount = 3u * contactCount;
    ArticulatedContactDiagnostics diagnostics = diagnosticsFor(
        articulated.articulationIndex,
        contactCount,
        articulated.nv
    );
    if (!operatorStructurallyValid(articulated) ||
        freeVelocity.size() != articulated.nv ||
        articulated.conic.contacts.size() != contactCount) {
        diagnostics.status =
            ArticulatedContactStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(freeVelocity)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }

    ContactSpaceConicProblem result;
    result.delassus = articulated.delassus;
    result.freeContactVelocity.assign(rowCount, 0.0);
    diagnostics = applyArticulatedContactJacobian(
        articulated,
        freeVelocity,
        result.freeContactVelocity
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    result.contacts.reserve(contactCount);
    for (const DenseContactBlock& source :
         articulated.conic.contacts) {
        if (!finite(source.targetVelocity) ||
            !finite(source.regularization) ||
            !finite(source.warmImpulse) ||
            !finite(source.friction) ||
            !(source.friction >= 0.0) ||
            !std::ranges::all_of(
                source.regularization,
                [](const double value) {
                    return finite(value) && value > 0.0;
                }
            )) {
            diagnostics.status =
                ArticulatedContactStatus::invalidContact;
            return diagnostics;
        }
        result.contacts.push_back({
            source.targetVelocity,
            source.regularization,
            source.warmImpulse,
            source.friction,
        });
    }
    contactSpace = std::move(result);
    return diagnostics;
}

ArticulatedContactDiagnostics applyArticulatedContactJacobian(
    const ArticulatedContactProblem& problem,
    const std::span<const double> generalizedVelocity,
    const std::span<double> contactVelocity
) {
    const std::size_t contactCount = problem.contactCount;
    const std::size_t rowCount = 3u * contactCount;
    const std::size_t nv = problem.nv;
    ArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            problem.articulationIndex,
            contactCount,
            nv
        );
    if (!operatorStructurallyValid(problem) ||
        generalizedVelocity.size() != nv ||
        contactVelocity.size() != rowCount) {
        diagnostics.status =
            ArticulatedContactStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(generalizedVelocity)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }
    std::vector<double> result(rowCount, 0.0);
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            result[row] +=
                problem.contactJacobian[row * nv + dof] *
                generalizedVelocity[dof];
        }
    }
    if (!finite(result)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(result, contactVelocity.begin());
    return diagnostics;
}

ArticulatedContactDiagnostics
applyArticulatedContactJacobianTranspose(
    const ArticulatedContactProblem& problem,
    const std::span<const double> contactImpulse,
    const std::span<double> generalizedImpulse
) {
    const std::size_t contactCount = problem.contactCount;
    const std::size_t rowCount = 3u * contactCount;
    const std::size_t nv = problem.nv;
    ArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            problem.articulationIndex,
            contactCount,
            nv
        );
    if (!operatorStructurallyValid(problem) ||
        contactImpulse.size() != rowCount ||
        generalizedImpulse.size() != nv) {
        diagnostics.status =
            ArticulatedContactStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(contactImpulse)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }
    std::vector<double> result(nv, 0.0);
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        for (std::size_t row = 0u; row < rowCount; ++row) {
            result[dof] +=
                problem.contactJacobian[row * nv + dof] *
                contactImpulse[row];
        }
    }
    if (!finite(result)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(result, generalizedImpulse.begin());
    return diagnostics;
}

ArticulatedContactDiagnostics
computeArticulatedContactImpulseResponse(
    const ArticulatedContactProblem& problem,
    const std::span<const double> impulses,
    const std::span<double> generalizedVelocityDelta,
    const std::span<double> contactVelocityDelta
) {
    const std::size_t contactCount = problem.contactCount;
    const std::size_t rowCount = 3u * contactCount;
    const std::size_t nv = problem.nv;
    ArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            problem.articulationIndex,
            contactCount,
            nv
        );
    if (!operatorStructurallyValid(problem) ||
        impulses.size() != rowCount ||
        generalizedVelocityDelta.size() != nv ||
        contactVelocityDelta.size() != rowCount) {
        diagnostics.status =
            ArticulatedContactStatus::invalidDimensions;
        return diagnostics;
    }
    if (!finite(impulses)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }

    std::vector<double> generalizedImpulse(nv, 0.0);
    const ArticulatedContactDiagnostics transposeDiagnostics =
        applyArticulatedContactJacobianTranspose(
            problem,
            impulses,
            generalizedImpulse
        );
    if (!transposeDiagnostics.succeeded()) {
        return transposeDiagnostics;
    }
    const CholeskyFactor factor = retainedFactor(problem);
    std::vector<double> deltaVelocity;
    if (!solve(factor, generalizedImpulse, deltaVelocity)) {
        diagnostics.status =
            ArticulatedContactStatus::factorizationFailure;
        return diagnostics;
    }
    diagnostics.minimumCholeskyPivot = factor.minimumPivot;
    diagnostics.maximumCholeskyPivot = factor.maximumPivot;

    std::vector<double> deltaContact(rowCount, 0.0);
    const ArticulatedContactDiagnostics jacobianDiagnostics =
        applyArticulatedContactJacobian(
            problem,
            deltaVelocity,
            deltaContact
        );
    if (!jacobianDiagnostics.succeeded()) {
        return jacobianDiagnostics;
    }
    double actionScale = 1.0;
    for (std::size_t row = 0u; row < rowCount; ++row) {
        double delassusAction = 0.0;
        for (std::size_t column = 0u;
             column < rowCount;
             ++column) {
            delassusAction +=
                problem.delassus[row * rowCount + column] *
                impulses[column];
        }
        actionScale = std::max({
            actionScale,
            std::abs(deltaContact[row]),
            std::abs(delassusAction),
        });
        diagnostics.maximumActionResidual = std::max(
            diagnostics.maximumActionResidual,
            std::abs(deltaContact[row] - delassusAction)
        );
    }
    const double actionTolerance =
        512.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max({nv, rowCount, std::size_t{1u}})) *
        actionScale;
    if (!finite(deltaVelocity) || !finite(deltaContact) ||
        !finite(diagnostics.maximumActionResidual) ||
        diagnostics.maximumActionResidual > actionTolerance) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(
        deltaVelocity,
        generalizedVelocityDelta.begin()
    );
    std::ranges::copy(
        deltaContact,
        contactVelocityDelta.begin()
    );
    return diagnostics;
}

ArticulatedContactDiagnostics applyArticulatedContactImpulses(
    const ArticulatedContactProblem& problem,
    const std::span<const double> impulses,
    const std::span<double> generalizedVelocity
) {
    const std::size_t contactCount = problem.contactCount;
    const std::size_t rowCount = 3u * contactCount;
    const std::size_t nv = problem.nv;
    ArticulatedContactDiagnostics diagnostics =
        diagnosticsFor(
            problem.articulationIndex,
            contactCount,
            nv
        );
    if (generalizedVelocity.size() != nv ||
        !finite(generalizedVelocity)) {
        diagnostics.status =
            generalizedVelocity.size() != nv
            ? ArticulatedContactStatus::invalidDimensions
            : ArticulatedContactStatus::nonfiniteInput;
        return diagnostics;
    }
    std::vector<double> deltaVelocity(nv, 0.0);
    std::vector<double> deltaContact(rowCount, 0.0);
    diagnostics = computeArticulatedContactImpulseResponse(
        problem,
        impulses,
        deltaVelocity,
        deltaContact
    );
    if (!diagnostics.succeeded()) {
        return diagnostics;
    }
    std::vector<double> candidate(
        generalizedVelocity.begin(),
        generalizedVelocity.end()
    );
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        candidate[dof] += deltaVelocity[dof];
    }
    if (!finite(candidate)) {
        diagnostics.status =
            ArticulatedContactStatus::nonfiniteResult;
        return diagnostics;
    }
    std::ranges::copy(candidate, generalizedVelocity.begin());
    return diagnostics;
}

} // namespace metalrobo
