# Numi RoomPlan

A small iPhone/iPad companion app for capturing an entire property as a
scene-authoring input for Numi Lab. It uses Apple's `RoomCaptureView`, keeps
each completed room, and combines them with `StructureBuilder` into one
property-scale parametric USDZ file.

## Run

```sh
cd examples/NumiRoomPlan
xcodegen generate
open NumiRoomPlan.xcodeproj
```

Select a physical LiDAR-capable iPhone or iPad. RoomPlan is not supported by
the Simulator. Allow camera access, scan slowly through the property, tap
**Finish room** at each room boundary, then tap **Continue scan**. After the
last room, **Send to Numi Lab** combines the rooms and opens the normal iOS
share sheet with the property-scale parametric USDZ file.

The app keeps the camera surface clear and uses only system-adaptive materials
for the small control group. The exported model remains in RoomPlan's
real-world units so dimensions stay correct when opened in Numi Lab or a USDZ
viewer. The app does not upload scans automatically.

## App Review notes

- RoomPlan requires a LiDAR-capable iPhone or iPad; the app presents a clear
  explanation on unsupported devices.
- Camera and LiDAR data is processed on-device. No network service, analytics,
  tracking, photo library, contacts, microphone, or location access is used.
- **Send to Numi Lab** invokes Apple's standard share sheet. The user chooses
  the destination and explicitly controls the USDZ share.
- Add the production privacy-policy URL to App Store Connect metadata before
  submission and set the same HTTPS URL in the `PRIVACY_POLICY_URL` build
  setting; the in-app Privacy button opens it when configured and otherwise
  explains the current data flow.

RoomPlan is an iPhone/iPad LiDAR framework, not a native visionOS scanning
framework. The iOS app can be offered as a compatible app on Apple Vision Pro,
but `RoomCaptureSession.isSupported` must remain the gate and scanning is
gracefully unavailable there. The clear camera-first UI and exported USDZ are
kept platform-friendly for a future native spatial viewer.

The app's RoomPlan behavior follows Apple's API documentation:
<https://developer.apple.com/augmented-reality/roomplan/>.
