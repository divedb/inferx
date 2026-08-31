#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "absl/status/status.h"
#include "gtest/gtest.h"
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

TEST(M2DTypeTest, EveryValueAndInvalidValueAreChecked) {
  constexpr std::array<DType, 13> kDTypes{
      DType::kBool,     DType::kUInt8,   DType::kInt8,   DType::kUInt16, DType::kInt16,
      DType::kUInt32,   DType::kInt32,   DType::kUInt64, DType::kInt64,  DType::kFloat16,
      DType::kBFloat16, DType::kFloat32, DType::kFloat64};
  for (DType dtype : kDTypes) {
    EXPECT_TRUE(DTypeSize(dtype).ok());
    EXPECT_TRUE(DTypeName(dtype).ok());
    EXPECT_TRUE(IsIntegral(dtype).ok());
    EXPECT_TRUE(IsFloatingPoint(dtype).ok());
  }
  // Deliberately constructs a wire-level invalid value to verify rejection.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const DType invalid_dtype = static_cast<DType>(255);
  EXPECT_EQ(DTypeSize(invalid_dtype).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(M2ShapeTest, ScalarEmptyRankAndOverflowContracts) {
  const Shape scalar = Shape::Create({}).value();
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.NumElements().value(), 1);

  constexpr std::array<uint64_t, 3> kEmpty{2, 0, 9};
  const Shape empty = Shape::Create(kEmpty).value();
  EXPECT_EQ(empty.NumElements().value(), 0);
  EXPECT_EQ(empty.Bytes(DType::kFloat64)->value(), 0);

  constexpr std::array<uint64_t, 9> kTooMany{};
  EXPECT_EQ(Shape::Create(kTooMany).status().code(), absl::StatusCode::kInvalidArgument);
  constexpr std::array<uint64_t, 2> kOverflow{std::numeric_limits<uint64_t>::max(), 2};
  EXPECT_EQ(Shape::Create(kOverflow)->NumElements().status().code(), absl::StatusCode::kOutOfRange);

  constexpr std::array<uint64_t, 1> kMaximum{std::numeric_limits<uint64_t>::max()};
  const Shape maximum = Shape::Create(kMaximum).value();
  EXPECT_EQ(maximum.NumElements().value(), kMaximum[0]);
  EXPECT_EQ(maximum.Bytes(DType::kFloat64).status().code(), absl::StatusCode::kOutOfRange);
  constexpr std::array<uint64_t, 8> kRankEight{1, 2, 1, 2, 1, 2, 1, 2};
  const Shape rank_eight = Shape::Create(kRankEight).value();
  EXPECT_EQ(rank_eight.rank(), 8);
  EXPECT_EQ(rank_eight.NumElements().value(), 16);
  EXPECT_EQ(rank_eight, Shape::Create(kRankEight).value());
}

TEST(M2StrideTest, ClassifiesContiguousPaddedAndOverlapping) {
  constexpr std::array<uint64_t, 2> kDims{2, 3};
  const Shape shape = Shape::Create(kDims).value();
  const Strides contiguous = Strides::Contiguous(shape).value();
  const LayoutAnalysis contiguous_layout = AnalyzeLayout(shape, contiguous, DType::kUInt8).value();
  EXPECT_TRUE(contiguous_layout.contiguous);
  EXPECT_TRUE(contiguous_layout.dense);
  EXPECT_EQ(contiguous_layout.overlap, OverlapKind::kNonOverlapping);

  constexpr std::array<uint64_t, 2> kPadded{8, 1};
  const LayoutAnalysis padded =
      AnalyzeLayout(shape, Strides::CreateElements(kPadded).value(), DType::kUInt8).value();
  EXPECT_FALSE(padded.contiguous);
  EXPECT_FALSE(padded.dense);
  EXPECT_EQ(padded.reachable_bytes.value(), 11);

  constexpr std::array<uint64_t, 2> kOverlapping{1, 1};
  const LayoutAnalysis overlapping =
      AnalyzeLayout(shape, Strides::CreateElements(kOverlapping).value(), DType::kUInt8).value();
  EXPECT_EQ(overlapping.overlap, OverlapKind::kMayOverlap);
}

TEST(M2BufferTest, MoveSubviewAlignmentHostAccessAndRelease) {
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

TEST(M2BufferTest, ZeroBytesAndAllocatorDomainLifetime) {
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

TEST(M2TensorTest, StridedCopySlicePermutationAndOverlapRules) {
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
  const TensorView source = TensorView::Create(source_storage.AsConst(), DType::kUInt8, shape,
                                               Strides::CreateElements(kSourceStrides).value())
                                .value();
  const MutableTensorView destination =
      MutableTensorView::Create(destination_storage, DType::kUInt8, shape,
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
  EXPECT_EQ(MutableTensorView::Create(destination_storage, DType::kUInt8, shape,
                                      Strides::CreateElements(kOverlap).value())
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_TRUE(source_buffer.Release().ok());
  EXPECT_TRUE(destination_buffer.Release().ok());
}

TEST(M2TensorTest, EmptyTensorMayEndAtBufferBoundary) {
  CpuAllocator allocator;
  Buffer buffer = allocator.Allocate(HostRequest(16)).value();
  constexpr std::array<uint64_t, 1> kDims{0};
  const Shape shape = Shape::Create(kDims).value();
  const Strides strides = Strides::Contiguous(shape).value();
  EXPECT_TRUE(TensorView::Create(buffer.View(ByteRange{ByteCount(0), ByteCount(16)}).value(),
                                 DType::kFloat32, shape, strides, ByteCount(16))
                  .ok());
  EXPECT_TRUE(buffer.Release().ok());
}

}  // namespace
}  // namespace inferx
