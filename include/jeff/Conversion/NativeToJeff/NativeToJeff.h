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

void populateNativeToJeffConversionPatterns(RewritePatternSet& patterns);

/// Add array-constant patterns that use the converter's result type.
///
/// The converted type must be a static rank-one tensor with the same element
/// count and a jeff element type. Integer bit patterns may be zero-extended to
/// a wider signless type; truncation and floating-point conversion are not
/// supported. Elements retain row-major order. The converter may use the source
/// value to restrict where this representation change applies.
///
/// These patterns take precedence over the constants registered by
/// populateNativeToJeffConversionPatterns. They do not change the conversion of
/// scalar constants or other arithmetic operations.
void populateNativeToJeffArrayConstantConversionPatterns(const TypeConverter& typeConverter,
                                                         RewritePatternSet& patterns);

} // namespace jeff

} // namespace mlir
