set(NUMI_IOS_PROJECT_DIR "" CACHE PATH "Compiled Bench Pendulum project from the Mac build")
if(NOT NUMI_IOS_PROJECT_DIR)
    return()
endif()
if(NOT METALROBO_HAS_SWIFT)
    message(FATAL_ERROR "The iOS app requires the Swift compiler.")
endif()

set(ios_resources
    "${METALROBO_METALLIB}"
    "${NUMI_IOS_PROJECT_DIR}/robot.worldpack"
    "${NUMI_IOS_PROJECT_DIR}/robot.taskpack"
    "${NUMI_IOS_PROJECT_DIR}/robot.actuators"
    "${NUMI_IOS_PROJECT_DIR}/robot.sensors"
    "${NUMI_IOS_PROJECT_DIR}/robot.reality"
)
set_source_files_properties("${METALROBO_METALLIB}" PROPERTIES GENERATED TRUE)
set_source_files_properties(${ios_resources} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
add_executable(numi_lab_ios MACOSX_BUNDLE
    apps/NumiLabIOS.swift
    bindings/swift/MetalRoboTaskRollout.swift
    ${ios_resources}
)
target_include_directories(numi_lab_ios PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo")
target_link_libraries(numi_lab_ios PRIVATE metalrobo)
target_compile_options(numi_lab_ios PRIVATE "$<$<COMPILE_LANGUAGE:Swift>:-parse-as-library>")
set_target_properties(numi_lab_ios PROPERTIES
    OUTPUT_NAME "Numi Lab"
    Swift_LANGUAGE_VERSION 6
    Swift_COMPILATION_MODE singlefile
    MACOSX_BUNDLE_BUNDLE_NAME "Numi Lab"
    MACOSX_BUNDLE_GUI_IDENTIFIER com.numilab.ios
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION 1
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/packaging/ios/Info.plist.in"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
)
