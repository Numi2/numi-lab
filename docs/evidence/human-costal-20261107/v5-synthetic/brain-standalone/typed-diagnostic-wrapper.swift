import Metal
import NumiBrainCore

// Isolated test-only exposure of the existing factory. This does not change
// the Brain runtime implementation, witness, transaction, or device kernels.
public extension MetalNumiBrainRuntime {
  @_spi(NumanXInterop) static func qualificationMakeRuntime(configuration: MetalNumiBrainConfiguration,
    publication: BrainParameterPublication, device: any MTLDevice) throws -> MetalNumiBrainRuntime {
    do { return try makeRuntime(configuration: configuration, publication: publication, device: device) }
    catch let error as TissueError {
      let message: String
      switch error {
      case .invalidGrid(let text), .invalidStructure(let text), .invalidConduction(let text),
           .invalidConnectome(let text), .invalidEvents(let text), .invalidParameters(let text),
           .invalidStimulus(let text), .transaction(let text), .metal(let text): message=text
      }
      fputs("brain-factory=TissueError: ",stderr)
      message.withCString { fputs($0,stderr) };fputs("\n",stderr)
      throw error
    } catch {fputs("brain-factory=other-error\n",stderr);throw error}
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
