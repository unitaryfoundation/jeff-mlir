cmake_minimum_required(VERSION 3.28...4.2)

project(
    jeff-mlir
    VERSION 0.3.0
    DESCRIPTION "A jeff dialect in MLIR"
    LANGUAGES CXX C
)

option(BUILD_JEFF_MLIR_DOCS "Build docs for the jeff-mlir project" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_JEFF_MLIR_TESTS "Build tests for the jeff-mlir project" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_JEFF_MLIR_TRANSLATION "Build the translation module of the jeff-mlir project" ON)
option(BUILD_JEFF_OPT "Build the jeff-opt executable" ${PROJECT_IS_TOP_LEVEL})

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD
        17
        CACHE STRING "C++ standard to conform to"
    )
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile commands" FORCE
)

include(cmake/SetupMLIR.cmake)
include(cmake/ExternalDependencies.cmake)

add_subdirectory(include)
add_subdirectory(lib)

if(BUILD_JEFF_MLIR_TESTS)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(unittests)
endif()
