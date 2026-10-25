if(APPLE)
    add_executable(numi-matter-human-skin-fat-static-query EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/human_skin_fat_static_query.cpp")
    target_include_directories(numi-matter-human-skin-fat-static-query PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../../apps"
        "${CMAKE_CURRENT_LIST_DIR}/../include")
    target_compile_features(numi-matter-human-skin-fat-static-query PRIVATE cxx_std_23)
    target_compile_options(numi-matter-human-skin-fat-static-query PRIVATE
        -Wall -Wextra -Wpedantic -Werror)
endif()
