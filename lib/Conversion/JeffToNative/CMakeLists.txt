file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(
    MLIRJeffToNative
    ${CONVERSION_SOURCES}
    DEPENDS
    JeffToNativeIncGen
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
