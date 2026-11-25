file(READ "${METAL_WORLD_SOURCE}" source)
# Derive default pipeline descriptors from the same kernel calls that load
# them at runtime. Specialized descriptors need their own script entries.
string(REGEX MATCHALL "createPipeline\\([ \t\r\n]*@\"mr_[a-z0-9_]+\"" calls "${source}")
set(pipelines)
foreach(call IN LISTS calls)
    string(REGEX MATCH "mr_[a-z0-9_]+" kernel "${call}")
    list(APPEND pipelines "    {\"compute_function\": \"alias:MetalRobo#${kernel}\"}")
endforeach()
list(REMOVE_DUPLICATES pipelines)
if(NOT pipelines)
    message(FATAL_ERROR "No MetalWorld pipelines found in ${METAL_WORLD_SOURCE}")
endif()
list(JOIN pipelines ",\n" entries)
file(WRITE "${PIPELINE_SCRIPT}"
    "{\n  \"libraries\": {\"paths\": [{\"label\": \"MetalRobo\", \"path\": \"./MetalRobo.metallib\"}]},\n"
    "  \"pipelines\": {\"compute_pipelines\": [\n${entries}\n  ]}\n}\n")
