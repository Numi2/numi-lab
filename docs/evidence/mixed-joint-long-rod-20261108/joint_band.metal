// The selected inverse holds a nine-scalar translation band and adjacent
// twist-edge entries. Every node pair across the two witnesses must fit.
inline bool mrResearchRodWitnessesFitSelectedBand(
    const uint firstA, const uint firstB, const uint firstEdge,
    const uint secondA, const uint secondB, const uint secondEdge
) {
    const uint low=min(min(firstA,firstB),min(secondA,secondB));
    const uint high=max(max(firstA,firstB),max(secondA,secondB));
    const uint lowEdge=min(firstEdge,secondEdge);
    const uint highEdge=max(firstEdge,secondEdge);
    return high-low<=2u && highEdge-lowEdge<=1u;
}
