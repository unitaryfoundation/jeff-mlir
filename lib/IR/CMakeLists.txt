add_mlir_dialect_library(
    MLIRJeff
    JeffDialect.cpp
    JeffInterfaces.cpp
    JeffOps.cpp
    DEPENDS
    MLIRJeffInterfacesIncGen
    MLIRJeffOpsIncGen
    MLIRJeffEnumsIncGen
)

target_include_directories(
    MLIRJeff
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>
)
