// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#pragma once
#include <mscclpp/gpu.hpp>
#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <registered_memory.hpp>

#include "mscclpp/gpu.hpp"

#include <connection.hpp>
using mscclpp::CudaIpcConnection;

namespace vortex::cuda {
  /*
   * runtime API Wrappers
   */
  struct MemoryRef;

  void HostMalloc(void **ptr, size_t size);
  void HostFree(void *ptr);
  void DeviceMalloc(void **ptr, size_t size);
  void DeviceFree(void *ptr);

  void MemcpyAsync(MemoryRef dst, MemoryRef src, std::shared_ptr<CudaIpcConnection> connection);
  // void MemsetAsync(MemoryRef dst, int value, cudaStream_t s);

  std::tuple<size_t, size_t> MemGetInfo();

  // --- Device
  int GetDevice();
  void SetDevice(int d);
  int GetDeviceCount();
  void DeviceEanblePeerAccess(int peer);
  bool DeviceCanAccessPeer(int device, int peer);
  cudaDeviceProp GetDeviceProperties(int device);

  void DeviceSynchronize();

  /*
   * Control Primitives Proxy
   */

  class DeviceGuard {
  private:
    int original_device_id;
    int current_device_id;

  public:
    explicit DeviceGuard(int id);
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    ~DeviceGuard();

    int device() const { return current_device_id; }
  };

  struct MemoryRef {
    // uint8_t *ptr = nullptr;
    size_t offset = 0, size = 0;
    int device = -1;
    std::shared_ptr<mscclpp::RegisteredMemory> origin;

    MemoryRef() = default;
    MemoryRef(size_t _offset, size_t _size, int _device, std::shared_ptr<mscclpp::RegisteredMemory> _origin) :
      offset(_offset), size(_size), device(_device), origin(_origin) {}

    MemoryRef(const mscclpp::RegisteredMemory &rhs) :
      offset(0), size(rhs.size()), device(rhs.deviceId()), origin(std::make_shared<mscclpp::RegisteredMemory>(rhs)) {}
    MemoryRef slice(size_t beg, size_t end) {return MemoryRef{offset + beg, end - beg, device, origin}; }
    MemoryRef slice_n(size_t beg, size_t bytes) {return slice(beg, beg + bytes); }

    uint8_t *ptr() {
      return origin ? static_cast<uint8_t*>(origin->data()) : nullptr;
    }

    template <typename T>
    operator T*() { return reinterpret_cast<T *>(origin->data()); }
    template <typename T>
    operator const T*() const { return reinterpret_cast<T *>(origin->data()); }
  };
  /*
   * Platform Information
   */
  class Platform {
    int deviceCount_ = 0;
    std::vector<cudaDeviceProp> deviceProps_;

    void EnableAllPeerAccess();  
    void warmUp();
  public:
    Platform();

    int deviceCount() { return deviceCount_; }
    cudaDeviceProp deviceProp(int device) { return deviceProps_[device]; }
  };
}