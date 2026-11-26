file(SHA256 "${SHADER_LIBRARY}" shader_hash)
file(SHA256 "${PIPELINE_SCRIPT}" pipeline_hash)
file(STRINGS "${RECEIPT}" recorded_inputs)
set(expected_inputs "${shader_hash}  MetalRobo.metallib;${pipeline_hash}  NumiLab.mtlp-json")
if(NOT recorded_inputs STREQUAL expected_inputs)
    message(FATAL_ERROR "G18G archive inputs changed. Rebake using the current shader library and pipeline script.")
endif()

execute_process(COMMAND xcrun -sdk macosx metal-lipo -archs "${DEVICE_ARCHIVE}"
    OUTPUT_VARIABLE device_architectures COMMAND_ERROR_IS_FATAL ANY)
# Older Metal toolchains print the GPU type and subtype numerically.
string(FIND "${device_architectures}" "unknown(16777235,451)" numeric_g18g)
if(NOT device_architectures MATCHES "(^|[ \r\n])applegpu_g18g([ \r\n]|$)" AND numeric_g18g EQUAL -1)
    message(FATAL_ERROR "Expected a G18G archive. Found architectures: ${device_architectures}")
endif()

get_filename_component(output_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${output_dir}")
execute_process(COMMAND xcrun -sdk macosx metal-lipo -create
    "${BASE_ARCHIVE}" "${DEVICE_ARCHIVE}" -output "${OUTPUT}.tmp"
    COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${OUTPUT}.tmp" "${OUTPUT}")
