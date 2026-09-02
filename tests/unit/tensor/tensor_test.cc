#include <absl/status/statusor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "inferx/base/token.h"
#include "inferx/tensor/allocator.h"
#include "inferx/tensor/buffer.h"
#include "inferx/tensor/device.h"
#include "inferx/tensor/dtype.h"
#include "inferx/tensor/shape.h"
#include "inferx/tensor/strides.h"
#include "inferx/tensor/tensor_view.h"

namespace inferx {
namespace {

AllocationRequest HostRequest(uint64_t bytes, uint64_t alignment = 64) {
  return AllocationRequest{Device::Host(), MemoryKind::kHost, ByteCount(bytes),
                           ByteCount(alignment), MemoryCategory::kTest};
}

TEST(DtypeTest, EveryValueAndInvalidValueAreChecked) {
  constexpr std::array<Dtype, 13> kDtypes{
      Dtype::kBool,     Dtype::kUInt8,   Dtype::kInt8,   Dtype::kUInt16, Dtype::kInt16,
      Dtype::kUInt32,   Dtype::kInt32,   Dtype::kUInt64, Dtype::kInt64,  Dtype::kFloat16,
      Dtype::kBFloat16, Dtype::kFloat32, Dtype::kFloat64};
  for (Dtype dtype : kDtypes) {
    EXPECT_TRUE(DtypeSize(dtype).ok());
    EXPECT_TRUE(DtypeName(dtype).ok());
    EXPECT_TRUE(IsIntegral(dtype).ok());
    EXPECT_TRUE(IsFloatingPoint(dtype).ok());
  }
  // Deliberately constructs a wire-level invalid value to verify rejection.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const Dtype invalid_dtype = static_cast<Dtype>(255);
  EXPECT_EQ(DtypeSize(invalid_dtype).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ShapeTest, ScalarEmptyRankAndOverflowContracts) {
  const Shape scalar = Shape::Create({}).value();
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.NumElements().value(), 1);

  constexpr std::array<uint64_t, 3> kEmpty{2, 0, 9};
  const Shape empty = Shape::Create(kEmpty).value();
  EXPECT_EQ(empty.NumElements().value(), 0);
  EXPECT_EQ(empty.Bytes(Dtype::kFloat64)->value(), 0);

  constexpr std::array<uint64_t, 9> kTooMany{};
  EXPECT_EQ(Shape::Create(kTooMany).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array<uint64_t, 2> kOverflow{std::numeric_limits<uint64_t>::max(), 2};
  EXPECT_EQ(Shape::Create(kOverflow)->NumElements().status().code(), absl::StatusCode::kOutOfRange);

  constexpr std::array<uint64_t, 1> kMaximum{std::numeric_limits<uint64_t>::max()};
  const Shape maximum = Shape::Create(kMaximum).value();
  EXPECT_EQ(maximum.NumElements().value(), kMaximum[0]);
  EXPECT_EQ(maximum.Bytes(Dtype::kFloat64).status().code(), absl::StatusCode::kOutOfRange);
  constexpr std::array<uint64_t, 8> kRankEight{1, 2, 1, 2, 1, 2, 1, 2};
  const Shape rank_eight = Shape::Create(kRankEight).value();
  EXPECT_EQ(rank_eight.rank(), 8);
  EXPECT_EQ(rank_eight.NumElements().value(), 16);
  EXPECT_EQ(rank_eight, Shape::Create(kRankEight).value());
}

TEST(StrideTest, ClassifiesContiguousPaddedAndOverlapping) {
  constexpr std::array<uint64_t, 2> kDims{2, 3};
  const Shape shape = Shape::Create(kDims).value();
  const Strides contiguous = Strides::Contiguous(shape).value();
  const LayoutAnalysis contiguous_layout = AnalyzeLayout(shape, contiguous, Dtype::kUInt8).value();
  EXPECT_TRUE(contiguous_layout.contiguous);
  EXPECT_TRUE(contiguous_layout.dense);
  EXPECT_EQ(contiguous_layout.overlap, OverlapKind::kNonOverlapping);

  constexpr std::array<uint64_t, 2> kPadded{8, 1};
  const LayoutAnalysis padded =
      AnalyzeLayout(shape, Strides::CreateElements(kPadded).value(), Dtype::kUInt8).value();
  EXPECT_FALSE(padded.contiguous);
  EXPECT_FALSE(padded.dense);
  EXPECT_EQ(padded.reachable_bytes.value(), 11);

  constexpr std::array<uint64_t, 2> kOverlapping{1, 1};
  const LayoutAnalysis overlapping =
      AnalyzeLayout(shape, Strides::CreateElements(kOverlapping).value(), Dtype::kUInt8).value();
  EXPECT_EQ(overlapping.overlap, OverlapKind::kMayOverlap);
}

TEST(StrideTest, ZeroStrideRuleAndSliceOverflowAreChecked) {
  constexpr std::array<uint64_t, 1> kZeroStride{0};
  EXPECT_TRUE(AnalyzeLayout(Shape::Create(std::array<uint64_t, 1>{0}).value(),
                            Strides::CreateElements(kZeroStride).value(), Dtype::kUInt8)
                  .ok());
  EXPECT_TRUE(AnalyzeLayout(Shape::Create(std::array<uint64_t, 1>{1}).value(),
                            Strides::CreateElements(kZeroStride).value(), Dtype::kUInt8)
                  .ok());
  EXPECT_EQ(AnalyzeLayout(Shape::Create(std::array<uint64_t, 1>{2}).value(),
                          Strides::CreateElements(kZeroStride).value(), Dtype::kUInt8)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  constexpr std::array<uint64_t, 3> kHugeEmpty{std::numeric_limits<uint64_t>::max(),
                                               std::numeric_limits<uint64_t>::max(), 0};
  const Shape huge_empty = Shape::Create(kHugeEmpty).value();
  const Strides huge_empty_strides = Strides::Contiguous(huge_empty).value();
  const LayoutAnalysis huge_empty_layout =
      AnalyzeLayout(huge_empty, huge_empty_strides, Dtype::kFloat64).value();
  EXPECT_TRUE(huge_empty_layout.contiguous);
  EXPECT_EQ(huge_empty_layout.reachable_bytes.value(), 0);

  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(1)).value();
  constexpr std::array<uint64_t, 1> kOne{1};
  constexpr std::array<uint64_t, 1> kLargeStride{std::numeric_limits<uint64_t>::max() / 2 + 1};
  const TensorView view =
      TensorView::Create(buffer.View({ByteCount(0), ByteCount(1)}).value(), Dtype::kUInt8,
                         Shape::Create(kOne).value(), Strides::CreateElements(kLargeStride).value())
          .value();
  EXPECT_EQ(view.Slice(0, 0, 1, 2).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_TRUE(buffer.Release().ok());
}

TEST(BufferTest, MoveSubviewAlignmentHostAccessAndRelease) {
  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(256)).value();
  EXPECT_EQ(buffer.alignment().value(), 64);
  Buffer moved = std::move(buffer);
  EXPECT_TRUE(buffer.empty());  // NOLINT(bugprone-use-after-move): moved-from contract
  MutableBufferView view = moved.MutableView(ByteRange{ByteCount(8), ByteCount(64)}).value();
  EXPECT_EQ(view.alignment().value(), 8);
  EXPECT_EQ(view.HostBytes()->size(), 64);
  EXPECT_EQ(view.Subview(ByteRange{ByteCount(65), ByteCount(0)}).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_TRUE(moved.Release().ok());
  EXPECT_EQ(moved.Release().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(BufferTest, ZeroBytesAndAllocatorDomainLifetime) {
  Buffer buffer;
  {
    CpuAllocator allocator;
    buffer = allocator.Allocate(HostRequest(0, 128)).value();
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size().value(), 0);
    EXPECT_EQ(buffer.alignment().value(), 128);
    EXPECT_EQ(buffer.View({ByteCount(0), ByteCount(0)})->HostBytes()->size(), 0);
  }
  EXPECT_TRUE(buffer.Release().ok());
}

TEST(TensorTest, StridedCopySlicePermutationAndOverlapRules) {
  CpuAllocator allocator;
  Buffer source_buffer = allocator.Allocate(HostRequest(64)).value();
  Buffer destination_buffer = allocator.Allocate(HostRequest(64)).value();
  auto source_storage = source_buffer.MutableView(ByteRange{ByteCount(0), ByteCount(64)}).value();
  auto destination_storage =
      destination_buffer.MutableView(ByteRange{ByteCount(0), ByteCount(64)}).value();
  for (size_t index = 0; index < source_storage.HostBytes()->size(); ++index) {
    (*source_storage.HostBytes())[index] = static_cast<std::byte>(index);
  }

  constexpr std::array<uint64_t, 2> kDims{2, 3};
  constexpr std::array<uint64_t, 2> kSourceStrides{5, 1};
  constexpr std::array<uint64_t, 2> kDestinationStrides{1, 2};
  const Shape shape = Shape::Create(kDims).value();
  const TensorView source = TensorView::Create(source_storage.AsConst(), Dtype::kUInt8, shape,
                                               Strides::CreateElements(kSourceStrides).value())
                                .value();
  const MutableTensorView destination =
      MutableTensorView::Create(destination_storage, Dtype::kUInt8, shape,
                                Strides::CreateElements(kDestinationStrides).value())
          .value();
  EXPECT_TRUE(CopyTensorCpu(source, destination).ok());
  EXPECT_EQ((*destination_storage.HostBytes())[0], std::byte{0});
  EXPECT_EQ((*destination_storage.HostBytes())[2], std::byte{1});
  EXPECT_EQ((*destination_storage.HostBytes())[1], std::byte{5});

  constexpr std::array<uint8_t, 2> kAxes{1, 0};
  EXPECT_EQ(source.Permute(kAxes)->shape().dimensions()[0], 3);
  EXPECT_EQ(source.Slice(1, 0, 3, 2)->shape().dim(1), 2);
  EXPECT_EQ(source.Slice(2, 0, 1).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(source.Slice(1, 0, 3, 0).status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(source.Slice(1, 2, 1).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array<uint8_t, 2> kDuplicateAxes{0, 0};
  EXPECT_EQ(source.Permute(kDuplicateAxes).status().code(), absl::StatusCode::kInvalidArgument);

  constexpr std::array<uint64_t, 2> kOverlap{1, 1};
  EXPECT_EQ(MutableTensorView::Create(destination_storage, Dtype::kUInt8, shape,
                                      Strides::CreateElements(kOverlap).value())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_TRUE(source_buffer.Release().ok());
  EXPECT_TRUE(destination_buffer.Release().ok());
}

TEST(TensorTest, EmptyTensorMayEndAtBufferBoundary) {
  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(16)).value();
  constexpr std::array<uint64_t, 1> kDims{0};
  const Shape shape = Shape::Create(kDims).value();
  const Strides strides = Strides::Contiguous(shape).value();
  EXPECT_TRUE(TensorView::Create(buffer.View(ByteRange{ByteCount(0), ByteCount(16)}).value(),
                                 Dtype::kFloat32, shape, strides, ByteCount(16))
                  .ok());
  EXPECT_TRUE(buffer.Release().ok());
}

TEST(TensorTest, DtypeAlignmentOffsetAndReachableBoundsAreChecked) {
  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(32)).value();
  constexpr std::array<uint64_t, 1> kFour{4};
  const Shape shape = Shape::Create(kFour).value();
  const Strides strides = Strides::Contiguous(shape).value();
  const BufferView full = buffer.View({ByteCount(0), ByteCount(32)}).value();
  EXPECT_EQ(TensorView::Create(full, Dtype::kFloat32, shape, strides, ByteCount(2)).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TensorView::Create(buffer.View({ByteCount(2), ByteCount(16)}).value(), Dtype::kFloat32,
                               shape, strides)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TensorView::Create(full, Dtype::kFloat64, shape, strides, ByteCount(8)).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_TRUE(buffer.Release().ok());
}

TEST(TensorTest, SameAllocationUsesMemmoveOnlyForContiguousCopies) {
  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(32)).value();
  MutableBufferView storage = buffer.MutableView({ByteCount(0), ByteCount(32)}).value();
  for (size_t index = 0; index < storage.HostBytes()->size(); ++index) {
    (*storage.HostBytes())[index] = static_cast<std::byte>(index);
  }

  constexpr std::array<uint64_t, 1> kEight{8};
  const Shape contiguous_shape = Shape::Create(kEight).value();
  const Strides contiguous_strides = Strides::Contiguous(contiguous_shape).value();
  const TensorView contiguous_source =
      TensorView::Create(storage.AsConst(), Dtype::kUInt8, contiguous_shape, contiguous_strides)
          .value();
  const MutableTensorView contiguous_destination =
      MutableTensorView::Create(storage, Dtype::kUInt8, contiguous_shape, contiguous_strides,
                                ByteCount(2))
          .value();
  EXPECT_TRUE(CopyTensorCpu(contiguous_source, contiguous_destination).ok());
  for (size_t index = 0; index < 8; ++index) {
    EXPECT_EQ((*storage.HostBytes())[index + 2], static_cast<std::byte>(index));
  }

  constexpr std::array<uint64_t, 2> kShape{2, 2};
  constexpr std::array<uint64_t, 2> kStrides{3, 1};
  const Shape strided_shape = Shape::Create(kShape).value();
  const Strides strided = Strides::CreateElements(kStrides).value();
  const TensorView strided_source =
      TensorView::Create(storage.AsConst(), Dtype::kUInt8, strided_shape, strided).value();
  const MutableTensorView strided_destination =
      MutableTensorView::Create(storage, Dtype::kUInt8, strided_shape, strided, ByteCount(1))
          .value();
  EXPECT_EQ(CopyTensorCpu(strided_source, strided_destination).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(buffer.Release().ok());
}

}  // namespace
}  // namespace inferx
