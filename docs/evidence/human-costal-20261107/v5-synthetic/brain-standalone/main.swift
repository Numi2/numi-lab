import Foundation
import Metal
import NumiBrainCore
@_spi(NumanXInterop) import NumiBrainMetal

// Standalone adaptation of the retained NumiBrain joint-publication test.
// The bootstrap sensors and costal geometry are synthetic; witness, ACK,
// physical owner and publication use their actual device-resident services.
struct ProbeFailure: Error { let message: String }
func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() { throw ProbeFailure(message: message) }
}
func value(_ key: String) throws -> String {
  guard let v=ProcessInfo.processInfo.environment[key], !v.isEmpty else { throw ProbeFailure(message:"missing \(key)") }; return v
}
func identity(_ key: String) throws -> UInt64 {
  guard let v=UInt64(try value(key), radix:16) else { throw ProbeFailure(message:"invalid \(key)") }; return v
}
final class Latch<T: Sendable>: @unchecked Sendable {
  private let condition=NSCondition(); private var result: Result<T, Error>?
  func complete(_ v: Result<T,Error>) { condition.lock(); if result == nil {result=v;condition.broadcast()}; condition.unlock() }
  func wait(_ timeout: TimeInterval=60) throws -> T {
    condition.lock();defer { condition.unlock() };let deadline=Date(timeIntervalSinceNow:timeout)
    while result == nil && condition.wait(until:deadline) {}
    guard let result else {throw ProbeFailure(message:"callback timeout")};return try result.get()
  }
}
func point(_ device: any MTLDevice) throws -> MetalSharedEventPoint {
  guard let event=device.makeSharedEvent() else { throw ProbeFailure(message:"event allocation failed") }
  return try MetalSharedEventPoint(event:event,value:1)
}
func bootstrap(_ device: any MTLDevice, _ compiled: CompiledSpeciesTemplate,
               _ token: BrainJointTransactionToken) throws -> NumanXSensorPacketLease {
  let sensors=try compiled.species.senses.filter(\.enabled).map { topology in
    try check(token.committedTimestamp.rawValue >= UInt64(topology.latencyMicroseconds),"bootstrap timestamp")
    let count=Int(topology.receptorCount)*Int(topology.observationDimension)
    guard let values=device.makeBuffer(length:count*MemoryLayout<Float>.stride,options:[.storageModeShared,.hazardTrackingModeTracked]),
          let valid=device.makeBuffer(length:Int(topology.receptorCount)*MemoryLayout<UInt32>.stride,options:[.storageModeShared,.hazardTrackingModeTracked]) else { throw ProbeFailure(message:"sensor allocation") }
    values.contents().assumingMemoryBound(to:Float.self).initialize(repeating:topology.modality == .proprioception ? 0.25 : 0.5,count:count)
    valid.contents().assumingMemoryBound(to:UInt32.self).initialize(repeating:1,count:Int(topology.receptorCount))
    return try MetalRawSensorBufferLease(buffer:values,modality:topology.modality,
      receptorTimestamp:BrainTimestamp(microseconds:token.committedTimestamp.rawValue-UInt64(topology.latencyMicroseconds)),
      receptorCount:topology.receptorCount,featureDimension:topology.observationDimension,validityBuffer:valid)
  }
  return try NumanXSensorPacketLease(transaction:token,compiledSpeciesTemplate:compiled,rawSensors:sensors)
}
func waitPreflight(_ ticket: MetalNumiBrainRuntime.NumanXPreparedControlTicket) throws {
  let deadline=Date(timeIntervalSinceNow:10);var status=ticket.status
  while status == .numanXPrepareSubmitted && Date()<deadline { Thread.sleep(forTimeInterval:0.001);status=ticket.status }
  try check(status == .numanXPreflightReady, ticket.preflightFailureDescription ?? "preflight \(status)")
}
func waitCommit(_ ticket: MetalNumiBrainRuntime.NumanXPreparedControlTicket) throws {
  let deadline=Date(timeIntervalSinceNow:10);var status=ticket.status
  while status != .committed && status != .aborted && status != .terminalQuarantined && status != .numanXAppliedValidationRetryRequired && Date()<deadline {Thread.sleep(forTimeInterval:0.001);status=ticket.status}
  try check(status == .committed,"joint commit status \(status)")
}
func run() throws {
  guard let device=MTLCreateSystemDefaultDevice() else { throw ProbeFailure(message:"Metal unavailable") }
  let equality=try MetalNumanXBridgeV1Runtime.SourceJointEqualities(payloadPath:value("NUMANX_JOINT_EQUALITIES"),fingerprint:identity("NUMANX_JOINT_EQUALITY_FP"))
  let tissue=try MetalNumanXBridgeV1Runtime.CostalTissueOwnership(cartilagePayloadPath:value("NUMANX_COSTAL_CARTILAGE"),bindingPayloadPath:value("NUMANX_COSTAL_BINDING"),bindingFingerprint:identity("NUMANX_COSTAL_BINDING_FP"))
  let world=try MetalNumanXBridgeV1Runtime.AuthoredMatterWorld(packagePath:value("NUMANX_MATTER_WORLD_PACKAGE"),humanSourceFingerprint:identity("NUMANX_HUMAN_SOURCE_FP"),worldFingerprint:identity("NUMANX_MATTER_WORLD_FP"),sourceJointEqualities:equality,costalTissueOwnership:tissue)
  let native=try MetalNumanXBridgeV1Runtime(libraryPath:value("NUMANX_METALROBO_LIBRARY"),device:device,configuration:.init(
    rigidPayloadPath:value("NUMANX_FULLBODY_RIGID"),musclePayloadPath:value("NUMANX_FULLBODY_MUSCLE"),supportContactPayloadPath:value("NUMANX_FULLBODY_CONTACT"),visualPackPath:value("NUMANX_FULLBODY_VISUAL_PACK"),visionProfilePath:value("NUMANX_FULLBODY_VISION_PROFILE"),metalRoboMetallibPath:value("NUMANX_METALROBO_METALLIB"),matterMetallibPath:value("NUMANX_MATTER_METALLIB"),matterMaterialPath:"",timestepMicroseconds:10,maximumRetainedBytes:2<<30,transactionSlotCount:2,authoredMatterWorld:world))
  let expectedIdentity=try identity("NUMANX_CONSTRAINED_HUMAN_SOURCE_FP")
  try check(native.info.modelSourceFingerprint == expectedIdentity,"model identity")
  let admittedWorld=try native.currentWorldInfo()
  try check(admittedWorld.femAttachmentCount == 448,"attachment count")
  let parameters=TissueParameters.corticalSheetV0
  let compiled=try NumanXFullBodyTransportTemplate.compile(latencyMicroseconds:10)
  let publication=try BrainParameterPublication.developmentalSeedV1(species:compiled.species,tissueParameters:parameters)
  let initial=try CPUTissueDynamics.makeRestingGrid(width:8,height:8,parameters:parameters)
  let configuration=MetalNumiBrainConfiguration(initialTissueState:initial,tissueParameters:parameters,tissueStimulus:.none,compiledSpeciesTemplate:compiled,randomContext:TissueRandomContext(seed:0x4e55_4d49,environmentIdentifier:0,episodeIdentifier:1),schedulerEnvironmentIdentifier:0,maximumEncodedSubsteps:1)
  let brain=try MetalNumiBrainRuntime.qualificationMakeRuntime(configuration:configuration,publication:publication,device:device)
  let tx=try brain.beginControl(controlStepIdentifier:1,basePhysicsGeneration:0,committedTimestamp:BrainTimestamp(microseconds:1000),targetTimestamp:BrainTimestamp(microseconds:1010),cachedDecisionFingerprint:0x4e58_4445_4349_5349)
  let sensors=try bootstrap(device,compiled,tx.token)
  let decision=try brain.submitInferAndDecide(tx,numanXSensors:sensors,signal:point(device))
  let motor=try brain.submitNumanXMotorCandidate(decision,transaction:tx,candidateDurationMicroseconds:10,signal:point(device))
  let rootLatch=Latch<MetalNumanXBridgeV1PreparedRoot>()
  try native.beginPhysicalRoot(transaction:tx.token,motor:motor) { rootLatch.complete($0) }
  let physical=try rootLatch.wait()
  _ = try brain.finishNumanXMotorSubmission(motor,transaction:tx,timeoutMilliseconds:10_000)
  try check(physical.identity.transactionFingerprint == tx.token.fingerprint,"physical transaction identity")
  let fast=try brain.submitProvisionalAcceptedFastRoot(tx,waitFor:physical.physicalPreparedPoint,signal:point(device))
  let prepared=try brain.submitNumanXPreparedControl(tx,provisionalFast:fast,identity:physical.identity,acceptedPhysicsGate:physical.acceptedPhysicsGate,sensorCandidate:physical.sensorCandidate,culturePrepared:physical.culture,developmentalIntents:nil,signal:point(device),thenSignal:point(device))
  _ = try prepared.waitUntilBrainPrepareCompleted(timeoutMilliseconds:10_000);try waitPreflight(prepared)
  let proposalLatch=Latch<MetalNumanXHumanMatterProposalLease>()
  try physical.submitProposal(brain:prepared) { proposalLatch.complete($0) }
  let proposal=try proposalLatch.wait();try waitPreflight(prepared)
  let joint=try brain.numanXPreparedJointCommitFingerprint(for:prepared,identity:physical.identity)
  try physical.reserveApplication(brain:prepared)
  let ack=try brain.qualificationSubmitAck(prepared,proposal:proposal,signal:point(device))
  let applyLatch=Latch<MetalNumanXHumanMatterAppliedLease>()
  try physical.submitApply(ack:ack) { applyLatch.complete($0) }
  let applied=try applyLatch.wait();try check(applied.commandDisposition == .acceptedPendingPublication,"apply disposition")
  let resolution=try physical.makeResolution(proposal:proposal,applied:applied,jointCommitFingerprint:joint,brainGeneration:tx.token.shadowGeneration)
  _ = try brain.qualificationValidate(prepared,ack:ack,applied:applied,resolution:resolution,signal:point(device))
  try waitCommit(prepared);try check(!brain.hasOpenControl,"control remains open")
  guard let aggregate=try native.aggregateSnapshotIfAvailable() else {throw ProbeFailure(message:"missing aggregate")}
  try check(aggregate.publicationEpoch == 1 && aggregate.brainGeneration == brain.committedGeneration && aggregate.physicsGeneration == 1 && aggregate.sensorGeneration == 1 && aggregate.identity == physical.identity,"published aggregate")
  print("PASS synthetic v5 real-Brain accepted publication epoch=\(aggregate.publicationEpoch) physics=\(aggregate.physicsGeneration) brain=\(aggregate.brainGeneration) transaction=\(physical.identity.transactionFingerprint)")
}
do {try run()} catch {fputs("synthetic v5 publication FAILED: \(error)\n",stderr);exit(1)}
