add_mlir_library(
    MLIRJeffTranslation
    Deserialize.cpp
    Serialize.cpp
    DEPENDS
    MLIRJeffEnumsIncGen
    MLIRJeffInterfacesIncGen
    MLIRJeffOpsIncGen
    LINK_LIBS
    PRIVATE
    MLIRJeff
    MLIRFuncDialect
    PUBLIC
    CapnProto::capnp
    DISABLE_INSTALL
)

target_include_directories(
    MLIRJeffTranslation PUBLIC ${jeff_SOURCE_DIR}/impl/cpp/src/capnp
                               ${jeff_BINARY_DIR}/impl/cpp/src/capnp
)

target_sources(MLIRJeffTranslation PRIVATE ${jeff_SOURCE_DIR}/impl/cpp/src/capnp/jeff.capnp.c++)
