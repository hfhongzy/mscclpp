
#include "io-sched.h"

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include <iostream>

using namespace mscclpp;

int main() {
  static constexpr size_t NUM_GPU = 4;
  static constexpr size_t TARGET_GPU = 0;

  uint8_t byte_test = 32 | 8;

  uint32_t *cpuSrcMem = nullptr, *cpuDstMem = nullptr;
  uint32_t *gpuDstMem = nullptr, *gpuSrcMem = nullptr;

  TransportFlags transports(Transport::CudaIpc);
  auto context = std::make_shared<Context>();

  std::cout << "context created." << std::endl;

  // size_t onload_len = 1024 * 1000 * 100;
  // size_t offload_len = 2048 * 1000 * 100;
  size_t onload_len = 400 * 1000 * 100;
  size_t offload_len = 400 * 1000 * 100;
  // 160 x 10^6

  const size_t onload_size = onload_len * sizeof(uint32_t);
  const size_t offload_size = offload_len * sizeof(uint32_t);

  MSCCLPP_CUDATHROW(cudaMallocHost(&cpuSrcMem, onload_size));  // Pinned memory
  MSCCLPP_CUDATHROW(cudaMallocHost(&cpuDstMem, offload_size));

  for (size_t i = 0; i < onload_len; ++i) {
    cpuSrcMem[i] = onload_len - i;
  }
  for (size_t i = 0; i < offload_len; ++i) {
    cpuDstMem[i] = 0;
  }
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaMalloc(&gpuSrcMem, offload_size));
  MSCCLPP_CUDATHROW(cudaMalloc(&gpuDstMem, onload_size));
  MSCCLPP_CUDATHROW(cudaMemset(gpuSrcMem, byte_test, offload_size));
  MSCCLPP_CUDATHROW(cudaMemset(gpuDstMem, 0, onload_size));
  // Dst memories are initialized with 0, while src memories are all non-zeros.

  std::cout << "INFO: CPU & GPU memories initialized." << std::endl;

  RegisteredMemory cpuSrcRegMem = context->registerCpuMemory(cpuSrcMem, onload_size, transports);
  RegisteredMemory cpuDstRegMem = context->registerCpuMemory(cpuDstMem, offload_size, transports);
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  RegisteredMemory gpuSrcRegMem = context->registerMemory(gpuSrcMem, offload_size, transports);
  RegisteredMemory gpuDstRegMem = context->registerMemory(gpuDstMem, onload_size, transports);

  std::cout << "INFO: CPU & GPU memories registered." << std::endl;

  std::vector<Endpoint> ep;
  ep.push_back(std::move(context->createEndpoint(EndpointConfig(Transport::CudaIpc, DeviceType::CPU))));
  for (size_t i = 0; i < NUM_GPU; ++i) {
    ep.push_back(std::move(context->createEndpoint(EndpointConfig(Transport::CudaIpc, DeviceType::GPU))));
  }

  std::cout << "INFO: Endpoints created." << std::endl;

  constexpr int granularity = 20'000'000; // 20MB
  // constexpr int granularity = 2000; // 20MB
  vortex::sched::LoadBalancedExchange exchange(granularity, NUM_GPU, context, ep);

  std::cout << "INFO: Exchange primitive constructed." << std::endl;
  
  exchange.launch(gpuDstRegMem, cpuSrcRegMem, cpuDstRegMem, gpuSrcRegMem);

  std::cout << "INFO: Exchange launched." << std::endl;

  exchange.sync();

  std::cout << "INFO: Exchange.launch() finished." << std::endl;
  // validate

  uint32_t *h2d = new uint32_t[onload_size];
  uint32_t *d2h = new uint32_t[offload_size];

  cudaSetDevice(TARGET_GPU);
  cudaMemcpy(h2d, gpuDstMem, onload_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(d2h, gpuSrcMem, offload_size, cudaMemcpyDeviceToHost);
  std::cout << "Start comparing." << std::endl;
  for (size_t i = 0; i < onload_len; ++i) {
    if (h2d[i] != cpuSrcMem[i]) {
      std::cout << "[H2D] Mismatch at " << i << ": got " << h2d[i] << ", expected " << cpuSrcMem[i] << std::endl;
      return 1;
    }
  }
  for (size_t i = 0; i < offload_len; ++i) {
    if (d2h[i] != cpuDstMem[i]) {
      std::cout << "[D2H] Mismatch at " << i << ": got " << d2h[i] << ", expected " << cpuDstMem[i] << std::endl;
      return 1;
    }
  }
  std::cout << "Answer validated." << std::endl;

  // free
  cudaSetDevice(TARGET_GPU);
  cudaFree(gpuSrcMem);
  cudaFree(gpuDstMem);
  cudaFreeHost(cpuSrcMem);
  cudaFreeHost(cpuDstMem);
  delete []h2d;
  delete []d2h;
}