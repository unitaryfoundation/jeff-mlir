file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    MLIRNativeToJeff
    ${CONVERSION_SOURCES}
    DEPENDS
    NativeToJeffIncGen
    MLIRJeffEnumsIncGen
    MLIRJeffInterfacesIncGen
    MLIRJeffOpsIncGen
    LINK_LIBS
    MLIRArithDialect
    MLIRDialect
    MLIRJeff
    MLIRMathDialect
    MLIRTensorDialect
    MLIRTransforms
    DISABLE_INSTALL
)
