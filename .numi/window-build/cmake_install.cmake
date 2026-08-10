# Install script for directory: /Users/home/Documents/emergentnumilife/MetalRobo

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib/libmetalrobo.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmetalrobo.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmetalrobo.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "@loader_path/../lib"
      -add_rpath "@loader_path"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmetalrobo.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmetalrobo.dylib")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_bench")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_bench" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_bench")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      -add_rpath "@loader_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_bench")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_bench")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_world_pack")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_world_pack" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_world_pack")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_world_pack")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_world_pack")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_episode_compile")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_episode_compile" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_episode_compile")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_episode_compile")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_episode_compile")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_motion_compile")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_motion_compile" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_motion_compile")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      -add_rpath "@loader_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_motion_compile")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_motion_compile")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_visual_cook")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_visual_cook" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_visual_cook")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      -add_rpath "@loader_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_visual_cook")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_visual_cook")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_environment_cook")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_environment_cook" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_environment_cook")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      -add_rpath "@loader_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_environment_cook")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_environment_cook")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_robot_catalog")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_robot_catalog" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_robot_catalog")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      -add_rpath "@loader_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_robot_catalog")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_robot_catalog")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_task_rollout")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_rollout" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_rollout")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_rollout")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_rollout")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/bin/metalrobo_task_train")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_train" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_train")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_train")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -u -r "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/metalrobo_task_train")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES "/Users/home/Documents/emergentnumilife/MetalRobo/tools/numi")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/libexec/numi" TYPE PROGRAM FILES
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/codex"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/train"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/evaluate"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/foundation"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/motion"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/robots"
    "/Users/home/Documents/emergentnumilife/MetalRobo/numi/commands/world-model"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/plugins" TYPE DIRECTORY FILES "/Users/home/Documents/emergentnumilife/MetalRobo/plugins/numi-lab")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/.agents/plugins" TYPE FILE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.agents/plugins/marketplace.json")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/numi" TYPE FILE FILES
    "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/generated/numi/VERSION"
    "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/generated/numi/REVISION"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/numi/python" TYPE DIRECTORY FILES "/Users/home/Documents/emergentnumilife/MetalRobo/python/metalrobo" FILES_MATCHING REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE REGEX "/[^/]*\\.py$" REGEX "/py\\.typed$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/Users/home/Documents/emergentnumilife/MetalRobo/include/")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/metalrobo" TYPE FILE FILES "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/shaders/MetalRobo.metallib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/metalrobo/schemas" TYPE FILE FILES
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/capture_manifest.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/perception_provider.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/tactile_calibration_record.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/tactile_observation.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/tactile_policy_checkpoint.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/tactile_stream.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/robot_dataset_manifest.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/visual_episode_stream.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/visual_frame_batch.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/visual_scene_manifest_v3.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/pointworld_model_pack.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/pointworld_observation.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/pointworld_robot_flow_candidates.schema.json"
    "/Users/home/Documents/emergentnumilife/MetalRobo/schemas/pointworld_forecast.schema.json"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/metalrobo" TYPE FILE FILES
    "/Users/home/Documents/emergentnumilife/MetalRobo/THIRD_PARTY_NOTICES.md"
    "/Users/home/Documents/emergentnumilife/MetalRobo/licenses/CISST_LICENSE.txt"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/home/Documents/emergentnumilife/MetalRobo/.numi/window-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
