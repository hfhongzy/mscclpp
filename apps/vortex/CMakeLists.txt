# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS include/*)

if(MSCCLPP_USE_ROCM)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(mscclpp_vortex_obj OBJECT)
target_sources(mscclpp_vortex_obj PRIVATE ${SOURCES})
target_sources(mscclpp_vortex_obj PUBLIC FILE_SET HEADERS FILES ${HEADERS})
target_include_directories(mscclpp_vortex_obj PRIVATE include ${PROJECT_SOURCE_DIR}/src/include SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})

target_link_libraries(mscclpp_vortex_obj PRIVATE ${GPU_LIBRARIES} PUBLIC mscclpp_obj)
set_target_properties(mscclpp_vortex_obj PROPERTIES LINKER_LANGUAGE CXX POSITION_INDEPENDENT_CODE 1 VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
if(MSCCLPP_USE_CUDA)
    target_compile_definitions(mscclpp_vortex_obj PRIVATE MSCCLPP_USE_CUDA)
elseif(MSCCLPP_USE_ROCM)
    target_compile_definitions(mscclpp_vortex_obj PRIVATE MSCCLPP_USE_ROCM)
endif()
if(MSCCLPP_NPKIT_FLAGS)
    target_compile_definitions(mscclpp_vortex_obj PRIVATE ${MSCCLPP_NPKIT_FLAGS})
endif()
add_library(mscclpp_vortex SHARED)
target_link_libraries(mscclpp_vortex PUBLIC mscclpp_obj mscclpp_vortex_obj)
set_target_properties(mscclpp_vortex PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})
add_library(mscclpp_vortex_static STATIC)
target_link_libraries(mscclpp_vortex_static PUBLIC mscclpp_obj mscclpp_vortex_obj)
set_target_properties(mscclpp_vortex_static PROPERTIES VERSION ${MSCCLPP_VERSION} SOVERSION ${MSCCLPP_SOVERSION})

install(TARGETS mscclpp_vortex_obj
    FILE_SET HEADERS DESTINATION ${INSTALL_PREFIX}/include)
install(TARGETS mscclpp_vortex
    LIBRARY DESTINATION ${INSTALL_PREFIX}/lib)
install(TARGETS mscclpp_vortex_static
    ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib)

if(MSCCLPP_BUILD_TESTS)
    add_subdirectory(test)
endif()
