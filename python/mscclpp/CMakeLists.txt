# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
include(FetchContent)
FetchContent_Declare(nanobind GIT_REPOSITORY https://github.com/wjakob/nanobind.git GIT_TAG v1.4.0)
FetchContent_MakeAvailable(nanobind)

FetchContent_Declare(dlpack GIT_REPOSITORY https://github.com/dmlc/dlpack.git GIT_TAG v1.1)
FetchContent_MakeAvailable(dlpack)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)
nanobind_add_module(mscclpp_py ${SOURCES})
set_target_properties(mscclpp_py PROPERTIES OUTPUT_NAME _mscclpp)
target_link_libraries(mscclpp_py PRIVATE dlpack mscclpp_static ${GPU_LIBRARIES})
target_include_directories(mscclpp_py SYSTEM PRIVATE ${GPU_INCLUDE_DIRS})
install(TARGETS mscclpp_py LIBRARY DESTINATION .)
