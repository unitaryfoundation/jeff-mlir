#pragma once

#include <capnp/common.h>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <kj/common.h>
#include <kj/io.h>
#include <kj/string-tree.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <string>
#ifdef _WIN32
#include <utility>
#endif

namespace jeff::test {

inline std::string readJeffFileToText(llvm::StringRef path) {
    auto file = llvm::sys::fs::openNativeFileForRead(path);
    if (!file) {
        llvm::errs() << "Failed to open file: " << path << "\n";
        llvm::report_fatal_error("Could not open file");
    }

    capnp::MallocMessageBuilder message;
#ifdef _WIN32
    kj::AutoCloseHandle autoCloseHandle(*file);
    kj::HandleInputStream input(std::move(autoCloseHandle));
    capnp::readMessageCopy(input, message);
#else
    const kj::AutoCloseFd autoCloseFd(*file);
    capnp::readMessageCopyFromFd(autoCloseFd, message);
#endif

    const auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
}

inline std::string moduleTextFromBuffer(const kj::ArrayPtr<capnp::word>& buffer) {
    capnp::FlatArrayMessageReader message(buffer);
    const auto module = message.getRoot<jeff::Module>();
    return module.toString().flatten().cStr();
}

} // namespace jeff::test
