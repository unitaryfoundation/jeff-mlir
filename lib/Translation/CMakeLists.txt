# Cap'n Proto reports malformed data through KJ exceptions. Keep unwinding local
# to the translation library so callers can continue to build without exceptions.
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)

add_mlir_library(
    MLIRJeffTranslation
    Deserialize.cpp
    Metadata.cpp
    Serialize.cpp
    ${jeff_SOURCE_DIR}/impl/cpp/src/capnp/jeff.capnp.c++
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

if(MSVC)
    # LLVM's __has_feature fallback makes KJ misdetect MSVC exception support.
    target_compile_definitions(MLIRJeffTranslation PUBLIC KJ_NO_EXCEPTIONS=0)
endif()
