#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;
class TypeConverter;

#define GEN_PASS_DECL_NATIVETOJEFF
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

#define GEN_PASS_REGISTRATION
#include "jeff/Conversion/NativeToJeff/NativeToJeff.h.inc"

namespace jeff {

/**
 * @brief Populate conversion from native MLIR operations to jeff.
 *
 * Array constants use their source types unless arrayConstantTypeConverter is provided. The
 * converter receives the source value and selects a static rank-one tensor with unchanged element
 * count and native jeff element types. Elements keep row-major order. Signless integers retain or
 * increase their width by zero extension; float types stay unchanged. Other patterns use native
 * operand and result types.
 */
void populateNativeToJeffConversionPatterns(
    RewritePatternSet& patterns, const TypeConverter* arrayConstantTypeConverter = nullptr);

} // namespace jeff

} // namespace mlir
