// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <chrono>
#include <iostream>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "io-sched.h"

using namespace mscclpp;

vortex::cuda::Platform platform;                // Enable GPU Peer Access

size_t total_bytes = 4 * (size_t)1000'000'000;  // [host-to-device] + [device-to-host] bytes
double fraction = 0.5;                          // [host-to-device] / ([host-to-device] + [device-to-host])
constexpr size_t NUM_GPU = 4;
constexpr size_t TARGET_GPU = 2;
constexpr int granularity = 20'000'000;         // Default 20MB

void naive_launch(void *gpuDstRegMem, void *cpuSrcRegMem, void *cpuDstRegMem, void *gpuSrcRegMem, size_t size1,
                  size_t size2, cudaStream_t stream1, cudaStream_t stream2) {
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(gpuDstRegMem, cpuSrcRegMem, size1, cudaMemcpyDeviceToHost, stream1));
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(cpuDstRegMem, gpuSrcRegMem, size2, cudaMemcpyDeviceToHost, stream2));

  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream1));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream2));
}

int naive_test() {
  uint8_t byte_test = 32 | 8;

  uint32_t *cpuSrcMem = nullptr, *cpuDstMem = nullptr;
  uint32_t *gpuDstMem = nullptr, *gpuSrcMem = nullptr;

  size_t onload_len = int(total_bytes * fraction) / 4;
  size_t offload_len = int(total_bytes * (1 - fraction)) / 4;

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
  MSCCLPP_CUDATHROW(cudaMemset(gpuSrcMem, byte_test, offload_size));
  MSCCLPP_CUDATHROW(cudaMalloc(&gpuDstMem, onload_size));
  MSCCLPP_CUDATHROW(cudaMemset(gpuDstMem, 0, onload_size));

  cudaStream_t stream1, stream2;
  MSCCLPP_CUDATHROW(cudaStreamCreate(&stream1));
  MSCCLPP_CUDATHROW(cudaStreamCreate(&stream2));

  // warmup
  for (int T = 5; T--;)
    naive_launch(gpuDstMem, cpuSrcMem, cpuDstMem, gpuSrcMem, onload_size, offload_size, stream1, stream2);

  auto start = std::chrono::high_resolution_clock::now();

  naive_launch(gpuDstMem, cpuSrcMem, cpuDstMem, gpuSrcMem, onload_size, offload_size, stream1, stream2);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  MSCCLPP_CUDATHROW(cudaStreamDestroy(stream1));
  MSCCLPP_CUDATHROW(cudaStreamDestroy(stream2));

  std::cout << "[Naive] Time: " << duration.count() << " ms" << std::endl;
  std::cout << "[Naive] Bandwidth: " << int((onload_size + offload_size) / (double)duration.count() / 1e6) << " GB/s"
            << std::endl;
  // validate

  uint32_t *h2d = new uint32_t[onload_size];
  uint32_t *d2h = new uint32_t[offload_size];

  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaMemcpy(h2d, gpuDstMem, onload_size, cudaMemcpyDeviceToHost));
  MSCCLPP_CUDATHROW(cudaMemcpy(d2h, gpuSrcMem, offload_size, cudaMemcpyDeviceToHost));

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

  std::cout << "[Naive] Answer validated." << std::endl;
  // free
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaFree(gpuSrcMem));
  MSCCLPP_CUDATHROW(cudaFree(gpuDstMem));
  MSCCLPP_CUDATHROW(cudaFreeHost(cpuSrcMem));
  MSCCLPP_CUDATHROW(cudaFreeHost(cpuDstMem));
  delete[] h2d;
  delete[] d2h;
  return 0;
}

int vortex_test() {

  uint8_t byte_test = 32 | 8;

  uint32_t *cpuSrcMem = nullptr, *cpuDstMem = nullptr;
  uint32_t *gpuDstMem = nullptr, *gpuSrcMem = nullptr;

  TransportFlags transports(Transport::CudaIpc);
  auto context = std::make_shared<Context>();

  std::cerr << "context created." << std::endl;

  size_t onload_len = int(total_bytes * fraction) / 4;
  size_t offload_len = int(total_bytes * (1 - fraction)) / 4;

  const size_t onload_size = onload_len * sizeof(uint32_t);
  const size_t offload_size = offload_len * sizeof(uint32_t);

  MSCCLPP_CUDATHROW(cudaMallocHost(&cpuSrcMem, onload_size));  // pinned memory
  MSCCLPP_CUDATHROW(cudaMallocHost(&cpuDstMem, offload_size)); // pinned memory

  for (size_t i = 0; i < onload_len; ++i) {
    cpuSrcMem[i] = onload_len - i;
  }
  for (size_t i = 0; i < offload_len; ++i) {
    cpuDstMem[i] = 0;
  }
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaMalloc(&gpuSrcMem, offload_size));
  MSCCLPP_CUDATHROW(cudaMemset(gpuSrcMem, byte_test, offload_size));
  MSCCLPP_CUDATHROW(cudaMalloc(&gpuDstMem, onload_size));
  MSCCLPP_CUDATHROW(cudaMemset(gpuDstMem, 0, onload_size));

  std::cerr << "INFO: CPU & GPU memories initialized." << std::endl;

  RegisteredMemory cpuSrcRegMem = context->registerCpuMemory(cpuSrcMem, onload_size, transports);
  RegisteredMemory cpuDstRegMem = context->registerCpuMemory(cpuDstMem, offload_size, transports);
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  RegisteredMemory gpuSrcRegMem = context->registerMemory(gpuSrcMem, offload_size, transports);
  RegisteredMemory gpuDstRegMem = context->registerMemory(gpuDstMem, onload_size, transports);

  std::cerr << "INFO: CPU & GPU memories registered." << std::endl;

  std::vector<Endpoint> ep;
  ep.push_back(context->createEndpoint(EndpointConfig(Transport::CudaIpc, DeviceType::CPU)));
  for (size_t i = 0; i < NUM_GPU; ++i) {
    ep.push_back(context->createEndpoint(EndpointConfig(Transport::CudaIpc, DeviceType::GPU)));
  }

  std::cerr << "INFO: Endpoints created." << std::endl;

  vortex::sched::LoadBalancedExchange exchange(context, ep, TARGET_GPU, granularity);

  std::cerr << "INFO: Exchange primitive constructed." << std::endl;

  exchange.warmup(context);

  std::cerr << "INFO: Exchange warmuped." << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  exchange.launch(gpuDstRegMem, cpuSrcRegMem, cpuDstRegMem, gpuSrcRegMem);

  std::cerr << "INFO: Exchange launched." << std::endl;

  exchange.sync();

  std::cerr << "INFO: Exchange.launch() finished." << std::endl;
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "[Vortex] Time: " << duration.count() << " ms" << std::endl;
  std::cout << "[Vortex] Bandwidth: " << int((onload_size + offload_size) / (double)duration.count() / 1e6) << " GB/s"
            << std::endl;
  // validate

  uint32_t *h2d = new uint32_t[onload_size];
  uint32_t *d2h = new uint32_t[offload_size];

  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaMemcpy(h2d, gpuDstMem, onload_size, cudaMemcpyDeviceToHost));
  MSCCLPP_CUDATHROW(cudaMemcpy(d2h, gpuSrcMem, offload_size, cudaMemcpyDeviceToHost));

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
  std::cout << "[Vortex] Answer validated." << std::endl;

  // free
  MSCCLPP_CUDATHROW(cudaSetDevice(TARGET_GPU));
  MSCCLPP_CUDATHROW(cudaFree(gpuSrcMem));
  MSCCLPP_CUDATHROW(cudaFree(gpuDstMem));
  MSCCLPP_CUDATHROW(cudaFreeHost(cpuSrcMem));
  MSCCLPP_CUDATHROW(cudaFreeHost(cpuDstMem));

  delete[] h2d;
  delete[] d2h;
  return 0;
}

int main() {
  naive_test();
  vortex_test();
  return 0;
}