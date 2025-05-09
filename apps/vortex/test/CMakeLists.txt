# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(MPI)

add_executable(vortex_test vortex_test.cc)
target_link_libraries(vortex_test mscclpp mscclpp_vortex ${GPU_LIBRARIES} ${NUMA_LIBRARIES} Threads::Threads MPI::MPI_CXX)
if(IBVERBS_FOUND)
    target_link_libraries(vortex_test ${IBVERBS_LIBRARIES})
    target_compile_definitions(vortex_test PRIVATE USE_IBVERBS)
endif()
target_include_directories(vortex_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/include ${PROJECT_SOURCE_DIR}/apps/vortex/include)