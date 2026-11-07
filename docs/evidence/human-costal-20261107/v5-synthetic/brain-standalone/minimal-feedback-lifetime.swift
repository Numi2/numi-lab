import Foundation
import Metal

struct Snapshot { let gpuStartTime: Double; let gpuEndTime: Double }
final class FeedbackBox: @unchecked Sendable { var feedback: (any MTL4CommitFeedback)? }
final class Owner {
  let queue: any MTL4CommandQueue
  let allocator: any MTL4CommandAllocator
  let commandBuffer: any MTL4CommandBuffer
  let residencySet: any MTLResidencySet
  let source: any MTLBuffer
  let destination: any MTLBuffer
  init() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeMTL4CommandQueue(),
      let allocator = device.makeCommandAllocator(), let commandBuffer = device.makeCommandBuffer(),
      let source = device.makeBuffer(length: 4096, options: [.storageModeShared, .hazardTrackingModeTracked]),
      let destination = device.makeBuffer(length: 4096, options: [.storageModePrivate, .hazardTrackingModeTracked]) else { throw NSError(domain: "setup", code: 1) }
    let descriptor = MTLResidencySetDescriptor(); descriptor.initialCapacity = 2
    let residencySet = try device.makeResidencySet(descriptor: descriptor)
    residencySet.addAllocation(source); residencySet.addAllocation(destination)
    residencySet.commit(); residencySet.requestResidency()
    self.queue = queue; self.allocator = allocator; self.commandBuffer = commandBuffer
    self.residencySet = residencySet; self.source = source; self.destination = destination
  }
  func submit() throws -> Snapshot {
    allocator.reset(); commandBuffer.beginCommandBuffer(allocator: allocator)
    commandBuffer.useResidencySet(residencySet)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw NSError(domain: "encoder", code: 1) }
    encoder.copy(sourceBuffer: source, sourceOffset: 0, destinationBuffer: destination, destinationOffset: 0, size: 4096)
    encoder.endEncoding(); commandBuffer.endCommandBuffer()
    let semaphore = DispatchSemaphore(value: 0)
    let options = MTL4CommitOptions(); let box = FeedbackBox()
    options.addFeedbackHandler { feedback in box.feedback = feedback; semaphore.signal() }
    queue.commit([commandBuffer], options: options); semaphore.wait()
    guard let feedback = box.feedback else { throw NSError(domain: "feedback", code: 1) }
    if feedback.error != nil { throw NSError(domain: "gpu", code: 1) }
    fputs("minimal=feedback-success\n", stderr)
    let start = feedback.gpuStartTime; let end = feedback.gpuEndTime
    fputs("minimal=timing-read\n", stderr)
    return Snapshot(gpuStartTime: start, gpuEndTime: end)
  }
  func copy() throws {
    _ = try submit()
    fputs("minimal=submit-returned\n", stderr)
  }
}
do { let owner = try Owner(); try owner.copy(); fputs("minimal=copy-returned\n", stderr) }
catch { fputs("minimal=threw\n", stderr); exit(1) }
