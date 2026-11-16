#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <string>

namespace
{
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr
makeCloud(int point_count)
{
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
  cloud->resize(point_count);
  const int side = std::max(1, static_cast<int>(std::cbrt(point_count)));
  for (int i = 0; i < point_count; ++i)
  {
    const int x = i % side;
    const int y = (i / side) % side;
    const int z = i / (side * side);
    auto& point = cloud->points[i];
    point.x = static_cast<float>(x) * 0.01f;
    point.y = static_cast<float>(y) * 0.01f;
    point.z = static_cast<float>(z) * 0.01f;
    point.r = static_cast<std::uint8_t>((x * 37 + y * 11 + z * 3) & 0xFF);
    point.g = static_cast<std::uint8_t>((x * 7 + y * 53 + z * 5) & 0xFF);
    point.b = static_cast<std::uint8_t>((x * 3 + y * 13 + z * 71) & 0xFF);
    const float length = std::sqrt(static_cast<float>(x * x + y * y + z * z + 1));
    point.normal_x = static_cast<float>(x) / length;
    point.normal_y = static_cast<float>(y) / length;
    point.normal_z = static_cast<float>(z) / length;
    point.curvature = static_cast<float>((i % 17) * 0.01);
  }
  cloud->width = point_count;
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

std::string
encodeCloud(const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr& cloud, bool lossless)
{
  pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBNormal> encoder(
      pcl::io::MANUAL_CONFIGURATION, false, 0.001, 0.01, false, 100, true, 8);
  encoder.setLossless(lossless);
  std::stringstream compressed;
  encoder.encodePointCloud(cloud, compressed);
  return compressed.str();
}

void
BM_OctreeCompressionEncode(benchmark::State& state, bool lossless)
{
  const auto cloud = makeCloud(static_cast<int>(state.range(0)));
  for (auto _ : state)
  {
    const std::string compressed = encodeCloud(cloud, lossless);
    benchmark::DoNotOptimize(compressed.data());
    state.counters["compressed_bytes"] = static_cast<double>(compressed.size());
  }
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(cloud->size() * sizeof(pcl::PointXYZRGBNormal)));
}

void
BM_OctreeCompressionDecode(benchmark::State& state, bool lossless)
{
  const auto cloud = makeCloud(static_cast<int>(state.range(0)));
  const std::string compressed = encodeCloud(cloud, lossless);
  for (auto _ : state)
  {
    pcl::io::OctreePointCloudCompression<pcl::PointXYZRGBNormal> decoder;
    std::istringstream input(compressed);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr decoded = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
    decoder.decodePointCloud(input, decoded);
    benchmark::DoNotOptimize(decoded->points.data());
  }
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(cloud->size() * sizeof(pcl::PointXYZRGBNormal)));
  state.counters["compressed_bytes"] = static_cast<double>(compressed.size());
}

void
BM_OctreeCompressionLossyEncode(benchmark::State& state)
{
  BM_OctreeCompressionEncode(state, false);
}

void
BM_OctreeCompressionLosslessEncode(benchmark::State& state)
{
  BM_OctreeCompressionEncode(state, true);
}

void
BM_OctreeCompressionLossyDecode(benchmark::State& state)
{
  BM_OctreeCompressionDecode(state, false);
}

void
BM_OctreeCompressionLosslessDecode(benchmark::State& state)
{
  BM_OctreeCompressionDecode(state, true);
}
} // namespace

int
main(int argc, char** argv)
{
  benchmark::RegisterBenchmark("OctreeCompression/Lossy/Encode", BM_OctreeCompressionLossyEncode)
      ->RangeMultiplier(4)
      ->Range(1024, 65536)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("OctreeCompression/Lossless/Encode", BM_OctreeCompressionLosslessEncode)
      ->RangeMultiplier(4)
      ->Range(1024, 65536)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("OctreeCompression/Lossy/Decode", BM_OctreeCompressionLossyDecode)
      ->RangeMultiplier(4)
      ->Range(1024, 65536)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("OctreeCompression/Lossless/Decode", BM_OctreeCompressionLosslessDecode)
      ->RangeMultiplier(4)
      ->Range(1024, 65536)
      ->Unit(benchmark::kMillisecond);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
