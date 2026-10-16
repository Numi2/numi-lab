#!/usr/bin/env python3
from pathlib import Path
BASE=Path('/Users/home/cardboard-implementation-20261014/snapshots/source-portable-paper-reference-185/matter/src/metal/contact.metalinc')
OUT=Path(__file__).parent/'candidate/matter/src/metal/contact.metalinc'
text=BASE.read_text()
needle='inline float ipcFeatureMollifierDifferential('
start=text.index(needle)
brace=text.index('{',start)
depth=0; end=None
for i in range(brace,len(text)):
    if text[i]=='{': depth+=1
    elif text[i]=='}':
        depth-=1
        if depth==0:
            end=i+1; break
assert end is not None
old=text[start:end]
new='''inline float ipcFeatureMollifierDifferential(
    const float3 first,
    const float3 second,
    thread float3& firstGradient,
    thread float3& secondGradient
) {
    firstGradient = float3(0.0f);
    secondGradient = float3(0.0f);
    if (!all(isfinite(first)) || !all(isfinite(second))) {
        firstGradient = float3(NAN);
        secondGradient = float3(NAN);
        return NAN;
    }
    // Normalize each feature edge before measuring its angle. This keeps the
    // C1 mollifier dimensionless and independent of the mesh's length unit.
    // Exact zero-length features retain the existing degenerate value of zero.
    const float firstScale = max(max(abs(first.x), abs(first.y)), abs(first.z));
    const float secondScale = max(max(abs(second.x), abs(second.y)), abs(second.z));
    if (!(firstScale > 0.0f) || !(secondScale > 0.0f)) return 0.0f;
    const float3 firstScaled = first / firstScale;
    const float3 secondScaled = second / secondScale;
    const float firstScaledNorm = sqrt(dot(firstScaled, firstScaled));
    const float secondScaledNorm = sqrt(dot(secondScaled, secondScaled));
    if (!(firstScaledNorm > 0.0f) || !(secondScaledNorm > 0.0f) ||
        !isfinite(firstScaledNorm) || !isfinite(secondScaledNorm)) {
        firstGradient = float3(NAN);
        secondGradient = float3(NAN);
        return NAN;
    }
    const float3 firstUnit = firstScaled / firstScaledNorm;
    const float3 secondUnit = secondScaled / secondScaledNorm;
    const float3 sineVector = cross(firstUnit, secondUnit);
    const float sinSquared = max(dot(sineVector, sineVector), 0.0f);
    constexpr float relativeThreshold = 1.0e-3f;
    const float ratio = clamp(sinSquared / relativeThreshold, 0.0f, 1.0f);
    // C1 IPC mollifier: zero at parallel/antiparallel edges and unit with zero
    // slope once sin^2(theta) reaches the dimensionless threshold.
    const float value = ratio * (2.0f - ratio);
    if (!(ratio > 0.0f && ratio < 1.0f)) return value;

    // d(sin^2(theta))/d(first) and d(...)/d(second), by the chain rule
    // through each normalized feature vector.
    const float inverseFirstLength =
        (1.0f / firstScale) / firstScaledNorm;
    const float inverseSecondLength =
        (1.0f / secondScale) / secondScaledNorm;
    if (!isfinite(inverseFirstLength) || !isfinite(inverseSecondLength)) {
        firstGradient = float3(NAN);
        secondGradient = float3(NAN);
        return NAN;
    }
    const float derivative = 2.0f * (1.0f - ratio) / relativeThreshold;
    // Differentiate the cross-product form used for sin^2 directly, then
    // project through d(unit(x))/dx = (I - uu^T)/|x|.
    const float3 firstUnitGradient = 2.0f * cross(secondUnit, sineVector);
    const float3 secondUnitGradient = 2.0f * cross(sineVector, firstUnit);
    firstGradient = derivative * inverseFirstLength *
        (firstUnitGradient - dot(firstUnit, firstUnitGradient) * firstUnit);
    secondGradient = derivative * inverseSecondLength *
        (secondUnitGradient - dot(secondUnit, secondUnitGradient) * secondUnit);
    if (!all(isfinite(firstGradient)) || !all(isfinite(secondGradient))) {
        firstGradient = float3(NAN);
        secondGradient = float3(NAN);
        return NAN;
    }
    return value;
}'''
assert old.count('kGeometryFloor')==5
OUT.write_text(text[:start]+new+text[end:])
print('old_bytes',len(old.encode()),'new_bytes',len(new.encode()))
