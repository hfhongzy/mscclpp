// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <vortex-util.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <mutex>

namespace vortex::cuda {
  /*
   * runtime API Wrappers
   */

  void HostMalloc(void **ptr, size_t size) {
    MSCCLPP_CUDATHROW(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
  }

  void HostFree(void *ptr) { MSCCLPP_CUDATHROW(cudaFreeHost(ptr)); }

  void DeviceMalloc(void **ptr, size_t size) { MSCCLPP_CUDATHROW(cudaMalloc(ptr, size)); }

  void DeviceFree(void *ptr) { MSCCLPP_CUDATHROW(cudaFree(ptr)); }

  void MemcpyAsync(MemoryRef dst, MemoryRef src, std::shared_ptr<CudaIpcConnection> connection) {
    assert (dst.size == src.size && src.origin != nullptr && dst.origin != nullptr);
    connection->write(*(dst.origin), dst.offset, *(src.origin), src.offset, src.size);
  }

  void DeviceSynchronize() {
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  }

  int GetDevice() {
    int r;
    MSCCLPP_CUDATHROW(cudaGetDevice(&r));
    return r;
  }

  void SetDevice(int d) { MSCCLPP_CUDATHROW(cudaSetDevice(d)); }

  int GetDeviceCount() {
    int c;
    MSCCLPP_CUDATHROW(cudaGetDeviceCount(&c));
    return c;
  }

  void DeviceEnablePeerAccess(int peer) { MSCCLPP_CUDATHROW(cudaDeviceEnablePeerAccess(peer, 0)); }

  bool DeviceCanAccessPeer(int device, int peer) {
    int r;
    MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&r, device, peer));
    return r;
  }

  cudaDeviceProp GetDeviceProperties(int device) {
    cudaDeviceProp prop;
    MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&prop, device));
    return prop;
  }

  // --- Memory Primitive Proxy
  DeviceGuard::DeviceGuard(int id) : original_device_id(GetDevice()), current_device_id(id) {
    SetDevice(current_device_id);
  }

  DeviceGuard::~DeviceGuard() {
    SetDevice(original_device_id);
  }

  /*
   * Platform Information
   */
  void Platform::EnableAllPeerAccess() {
    for (int i = 0; i < deviceCount_; i++) {
      DeviceGuard on(i);
      for (int j = 0; j < deviceCount_; j++) {
        if (i == j) continue;
        int attr = 0;
        MSCCLPP_CUDATHROW(cudaDeviceGetP2PAttribute(&attr, cudaDevP2PAttrAccessSupported, i, j));

        if (attr) {
          std::cerr << "P2P is supported between " << i << " and " << j << std::endl;
          DeviceEnablePeerAccess(j);
        } else {
          std::cerr << "P2P is not supported between " << i << " and " << j << std::endl;
        }
      }
    }
  }

  Platform::Platform() : deviceCount_(GetDeviceCount()), deviceProps_(deviceCount_) {
    EnableAllPeerAccess();
    for (int i = 0; i < deviceCount_; i++) {
      deviceProps_[i] = GetDeviceProperties(i);
    }
  }

}
