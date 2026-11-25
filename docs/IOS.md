# Numi Lab on iOS and iPadOS

The iOS target runs the Bench Pendulum through the same native WorldPack,
task, actuator, and Metal solver used by the Mac app. It shows the measured
joint angle and supports live effort changes, pause, and reset. Its diagram
uses SwiftUI. The iOS build requires the iOS 26 SDK. Use the Mac workbench to
import models, edit experiments, and compile presentation packs.

## Build the simulator app

Build the Mac app resources first on the Mac. This prepares the shared fixture.

```sh
cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
cmake --build build/mac --target numi_lab_macos --parallel 3
cmake -S . -B build/ios-simulator -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphonesimulator -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
  '-DCMAKE_Swift_FLAGS=-target arm64-apple-ios26.0-simulator' \
  -DNUMI_IOS_PROJECT_DIR="$PWD/build/mac/generated/numi-mac/Scenes/Bench Pendulum.numiproject"
cmake --build build/ios-simulator --target numi_lab_ios --parallel 3
sim_stage=$(mktemp -d "${TMPDIR%/}/numi-simulator.XXXXXX")
ditto --norsrc "build/ios-simulator/Numi Lab.app" "$sim_stage/Numi Lab.app"
codesign --force --sign - "$sim_stage/Numi Lab.app"
xcrun simctl install booted "$sim_stage/Numi Lab.app"
xcrun simctl launch --console --terminate-running-process booted com.numilab.ios --self-check
```

The check advances 100 native control ticks, measures the excursion, and
verifies reset. Simulator builds use the buffered texture upload path.
Device testing and application distribution require a signed device build
and a physical iPhone or iPad.
