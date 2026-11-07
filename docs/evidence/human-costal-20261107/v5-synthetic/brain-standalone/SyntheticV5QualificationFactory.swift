import Metal
import NumiBrainCore

// Isolated test-only exposure of the existing factory. This does not change
// the Brain runtime implementation, witness, transaction, or device kernels.
public extension MetalNumiBrainRuntime {
  static func qualificationMakeRuntime(configuration: MetalNumiBrainConfiguration,
    publication: BrainParameterPublication, device: any MTLDevice) throws -> MetalNumiBrainRuntime {
    try makeRuntime(configuration: configuration, publication: publication, device: device)
  }
}
