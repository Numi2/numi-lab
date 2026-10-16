foreach(required IN ITEMS
        OUTPUT
        ENVIRONMENT_SOURCE
        PBR_SOURCE
        ASSET_COOK_SOURCE)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

file(SHA256 "${ENVIRONMENT_SOURCE}" environment_source_hash)
file(SHA256 "${PBR_SOURCE}" pbr_source_hash)
file(SHA256 "${ASSET_COOK_SOURCE}" asset_cook_source_hash)
string(
    SHA256
    environment_kernel_hash
    "${environment_source_hash}:${pbr_source_hash}"
)
string(
    SHA256
    asset_cook_kernel_hash
    "${asset_cook_source_hash}"
)

set(temporary "${OUTPUT}.tmp")
file(
    WRITE
    "${temporary}"
    "#pragma once\n"
    "#define METALROBO_ENVIRONMENT_KERNEL_HASH \"sha256:${environment_kernel_hash}\"\n"
    "#define METALROBO_VISUAL_ASSET_COOK_KERNEL_HASH \"sha256:${asset_cook_kernel_hash}\"\n"
)
file(RENAME "${temporary}" "${OUTPUT}")
