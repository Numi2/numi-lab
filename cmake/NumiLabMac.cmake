set(numi_mac_resources "${CMAKE_CURRENT_BINARY_DIR}/generated/numi-mac")
set(numi_mac_contents "Numi Lab.app/Contents")
set(NUMI_MAC_BUILD_NUMBER "1" CACHE STRING "Numi Lab Mac app build number")
configure_file(packaging/macos/Info.plist.in "${numi_mac_resources}/Info.plist" @ONLY)

option(NUMI_MAC_AOT_PIPELINES "Include precompiled Metal pipelines in Numi Lab.app" ON)
if(NUMI_MAC_AOT_PIPELINES)
    set(numi_mac_world_source "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/MetalWorld.mm")
    set(numi_mac_pipeline_script "${METALROBO_SHADER_DIR}/NumiLab.mtlp-json")

    set(numi_mac_pipeline_script_generator
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NumiLabMacPipelineScript.cmake")
    # Precompile the base GPU architecture in each generation. The retained IR
    # covers GPU models that need a different slice or an updated OS compiler.
    set(NUMI_MAC_GPU_ARCHITECTURES
        "applegpu_g13g;applegpu_g14g;applegpu_g15g;applegpu_g16g;applegpu_g17g"
        CACHE STRING "GPU architectures to precompile for the Numi Lab Mac app")
    string(SHA256 numi_mac_gpu_hash "${NUMI_MAC_GPU_ARCHITECTURES}")
    string(SUBSTRING "${numi_mac_gpu_hash}" 0 12 numi_mac_gpu_key)
    set(numi_mac_pipeline_archive "${METALROBO_SHADER_DIR}/${numi_mac_gpu_key}/MetalRobo.binary.metallib")
    set(numi_mac_gpu_flags)
    foreach(arch IN LISTS NUMI_MAC_GPU_ARCHITECTURES)
        list(APPEND numi_mac_gpu_flags -arch "${arch}")
    endforeach()
    add_custom_command(
        OUTPUT "${numi_mac_pipeline_archive}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${METALROBO_SHADER_DIR}/${numi_mac_gpu_key}"
        COMMAND "${CMAKE_COMMAND}" "-DMETAL_WORLD_SOURCE=${numi_mac_world_source}"
                "-DPIPELINE_SCRIPT=${numi_mac_pipeline_script}"
                -P "${numi_mac_pipeline_script_generator}"
        COMMAND xcrun -sdk macosx metal-tt "${numi_mac_pipeline_script}"
                -target air64-apple-macos26.0 ${numi_mac_gpu_flags}
                -compress-resources -no-pack-sources -o "${numi_mac_pipeline_archive}"
        DEPENDS "${METALROBO_METALLIB}" "${numi_mac_world_source}"
                "${numi_mac_pipeline_script_generator}"
        WORKING_DIRECTORY "${METALROBO_SHADER_DIR}"
        VERBATIM
    )

    set(numi_mac_optional_pipeline "${numi_mac_pipeline_archive}")
endif()

set(numi_mac_scene "${numi_mac_resources}/Scenes")
set(numi_mac_model "${CMAKE_CURRENT_SOURCE_DIR}/assets/window")
add_custom_command(
    OUTPUT "${numi_mac_scene}/catalog.version"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${numi_mac_scene}"
    COMMAND $<TARGET_FILE:metalrobo_numi_window_catalog_compile>
            --starter --workspace "${CMAKE_CURRENT_SOURCE_DIR}"
            --output "${numi_mac_scene}"
    COMMAND $<TARGET_FILE:metalrobo_numi_window_catalog_compile>
            --urdf "${numi_mac_model}/bench-pendulum.urdf"
            --project-id bench-pendulum --output "${numi_mac_scene}/Bench Pendulum.numiproject"
    DEPENDS metalrobo_numi_window_catalog_compile
            "${numi_mac_model}/bench-pendulum.urdf"
            "${numi_mac_model}/px4-x500-starter.urdf"
            "${numi_mac_model}/px4-x500.glb"
            "${numi_mac_model}/px4-x500-rotor-0.glb"
            "${numi_mac_model}/px4-x500-rotor-1.glb"
            "${numi_mac_model}/px4-x500-rotor-2.glb"
            "${numi_mac_model}/px4-x500-rotor-3.glb"
    VERBATIM
)

set(numi_mac_icon_source "${CMAKE_CURRENT_SOURCE_DIR}/plugins/numi-lab/assets/icon.png")
add_custom_command(
    OUTPUT "${numi_mac_resources}/NumiLab.icns"
    COMMAND /usr/bin/sips -s format icns "${numi_mac_icon_source}"
            --out "${numi_mac_resources}/NumiLab.icns"
    DEPENDS "${numi_mac_icon_source}"
    VERBATIM
)

add_custom_target(numi_lab_macos DEPENDS
    metalrobo_task_rollout
    ${numi_mac_optional_pipeline}
    "${numi_mac_scene}/catalog.version"
    "${numi_mac_resources}/NumiLab.icns"
)
install(TARGETS metalrobo_task_rollout
    RUNTIME DESTINATION "${numi_mac_contents}/MacOS"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
install(TARGETS metalrobo_numi_window_catalog_compile
    RUNTIME DESTINATION "${numi_mac_contents}/Helpers"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
install(TARGETS metalrobo
    LIBRARY DESTINATION "${numi_mac_contents}/Frameworks"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
install(FILES "${numi_mac_resources}/Info.plist"
    DESTINATION "${numi_mac_contents}"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
install(FILES "${METALROBO_METALLIB}" ${numi_mac_optional_pipeline}
    "${numi_mac_resources}/NumiLab.icns" THIRD_PARTY_NOTICES.md
    licenses/CISST_LICENSE.txt
    licenses/PX4_X500_BSD_LICENSE.txt
    examples/plastic_bag_franka/approved_20261024/source/LIBCCD-BSD-LICENSE.txt
    DESTINATION "${numi_mac_contents}/Resources"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
install(DIRECTORY "${numi_mac_scene}"
    DESTINATION "${numi_mac_contents}/Resources"
    COMPONENT NumiLabMac EXCLUDE_FROM_ALL)
