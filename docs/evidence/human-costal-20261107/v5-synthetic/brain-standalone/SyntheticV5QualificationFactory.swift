import Metal
import NumiBrainCore

// Isolated test-only exposure of the existing factory. This does not change
// the Brain runtime implementation, witness, transaction, or device kernels.
public extension MetalNumiBrainRuntime {
  @_spi(NumanXInterop) static func qualificationMakeRuntime(configuration: MetalNumiBrainConfiguration,
    publication: BrainParameterPublication, device: any MTLDevice) throws -> MetalNumiBrainRuntime {
    try makeRuntime(configuration: configuration, publication: publication, device: device)
  }
  @_spi(NumanXInterop) func qualificationSubmitAck(_ ticket: NumanXPreparedControlTicket,
    proposal: MetalNumanXHumanMatterProposalLease, signal: MetalSharedEventPoint) throws -> MetalNumanXHumanMatterBrainAckTicket {
    try submitNumanXBrainAck(ticket, proposal: proposal, signal: signal)
  }
  @_spi(NumanXInterop) func qualificationValidate(_ ticket: NumanXPreparedControlTicket,
    ack: MetalNumanXHumanMatterBrainAckTicket, applied: MetalNumanXHumanMatterAppliedLease,
    resolution: MetalNumanXJointResolutionReservation, signal: MetalSharedEventPoint) throws -> MetalNumanXHumanMatterAppliedValidationTicket {
    try validateNumanXAppliedRoot(ticket, ack: ack, applied: applied, resolution: resolution, signal: signal)
  }

}
