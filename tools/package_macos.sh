#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: tools/package_macos.sh OUTPUT_DIRECTORY' >&2
    exit 2
fi

repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
build=${NUMI_MAC_BUILD_DIR:-"$repo/.numi/build/macos"}
identity=${NUMI_MAC_SIGN_IDENTITY:-}
notary_profile=${NUMI_MAC_NOTARY_PROFILE:-}
if { [ -n "$identity" ] && [ -z "$notary_profile" ]; } ||
   { [ -z "$identity" ] && [ -n "$notary_profile" ]; }; then
    printf '%s\n' 'Set both NUMI_MAC_SIGN_IDENTITY and NUMI_MAC_NOTARY_PROFILE for notarized distribution.' >&2
    exit 2
fi

mkdir -p "$1"
output=$(CDPATH= cd "$1" && pwd -P)
stage=$(mktemp -d "${TMPDIR:-/tmp}/numi-macos.XXXXXX")
trap 'rm -rf "$stage"' EXIT
trap 'exit 1' HUP INT TERM

cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
cmake --build "$build" --target numi_lab_macos --parallel 4
cmake --install "$build" --prefix "$stage" --component NumiLabMac

app="$stage/Numi Lab.app"
library="$app/Contents/Frameworks/libmetalrobo.dylib"
catalog_compiler="$app/Contents/Helpers/metalrobo_numi_window_catalog_compile"
xattr -cr "$app"
[ "$(lipo -archs "$app/Contents/MacOS/metalrobo_task_rollout")" = arm64 ]
[ "$(lipo -archs "$library")" = arm64 ]
version=$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$app/Contents/Info.plist")
name="Numi-Lab-$version-macOS-arm64"
if [ -n "$identity" ]; then
    codesign --force --options runtime --timestamp --sign "$identity" "$library"
    codesign --force --options runtime --timestamp --sign "$identity" "$catalog_compiler"
    codesign --force --options runtime --timestamp --sign "$identity" "$app"
else
    name="$name-local-preview"
    codesign --force --sign - "$library"
    codesign --force --sign - "$catalog_compiler"
    codesign --force --sign - "$app"
fi
codesign --verify --deep --strict "$app"
"$repo/tests/numi_mac_workbench_test.sh" "$app"
archive="$stage/$name.zip"
ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"

if [ -n "$notary_profile" ]; then
    xcrun notarytool submit "$archive" --keychain-profile "$notary_profile" \
        --wait --output-format plist > "$stage/notary.plist"
    status=$(/usr/libexec/PlistBuddy -c 'Print status' "$stage/notary.plist")
    xcrun_id=$(/usr/libexec/PlistBuddy -c 'Print id' "$stage/notary.plist")
    if [ "$status" != Accepted ]; then
        printf 'Apple notarization %s: %s\n' "$status" "$xcrun_id" >&2
        exit 1
    fi
    xcrun stapler staple "$app"
    xcrun stapler validate "$app"
    spctl --assess --type execute "$app"
    rm "$archive"
    ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
fi

(
    cd "$stage"
    shasum -a 256 "$name.zip" > "$name.zip.sha256"
)
mv "$archive" "$stage/$name.zip.sha256" "$output/"
printf '%s\n' "$output/$name.zip"
