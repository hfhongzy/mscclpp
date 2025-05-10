#pragma once
#include <mscclpp/gpu.hpp>
#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <registered_memory.hpp>

#include <connection.hpp>
using mscclpp::CudaIpcConnection;

namespace vortex::cuda {
  /*
   * runtime API Wrappers
   */
  class MemoryRef;

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

  /*
   * Memory Primitive Proxy
   */

  // template<typename T>
  // struct HostAllocator {
  //   using value_type = T;

  //   HostAllocator() noexcept {}

  //   HostAllocator(const HostAllocator<T>&) noexcept {}

  //   T* allocate(std::size_t n) {
  //     T* ptr;
  //     HostMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(T));
  //     return ptr;
  //   }

  //   void deallocate(T* p, std::size_t) noexcept {
  //     HostFree(p);
  //   }
  // };

  // template <typename T>
  // using HostVector = std::vector<T, HostAllocator<T>>;

  // struct DeviceMemoryDeleter {
  //   void operator()(void* ptr) const {
  //     DeviceFree(ptr);
  //   }
  // };

  // class DeviceMemory {
  //   std::unique_ptr<void, DeviceMemoryDeleter> ptr_{nullptr};
  //   size_t size_{0};
  //   int device_{-1};
  // public:
  //   DeviceMemory() = default;
  //   DeviceMemory(size_t size);

  //   void *get() {return ptr_.get(); }
  //   const void *get() const {return ptr_.get(); }

  //   template <typename T>
  //   operator T*() { return reinterpret_cast<T *>(get()); }
  //   template <typename T>
  //   operator const T*() const { return reinterpret_cast<T *>(get()); }

  //   size_t size() const { return size_; }
  //   int device() const {return device_; }
  // };

  struct MemoryRef {
    // uint8_t *ptr = nullptr;
    size_t offset = 0, size = 0;
    int device = -1;
    std::shared_ptr<mscclpp::RegisteredMemory> origin;

    MemoryRef() = default;
    MemoryRef(size_t _offset, size_t _size, int _device, std::shared_ptr<mscclpp::RegisteredMemory> _origin) :
      offset(_offset), size(_size), device(_device), origin(_origin) {}

    // MemoryRef(int d): device(d) {}
    // template<typename T>
    // MemoryRef(HostVector<T> &rhs) : 
    //   ptr(reinterpret_cast<uint8_t *>(&rhs[0])), size(rhs.size() * sizeof(T)), device(-1) {}

    // MemoryRef(DeviceMemory &rhs) :
    //   offset(0), size(rhs.size()), device(rhs.device()), origin() {}

    MemoryRef(const mscclpp::RegisteredMemory &rhs) :
      offset(0), size(rhs.size()), device(rhs.deviceId()), origin(std::make_shared<mscclpp::RegisteredMemory>(rhs)) {}

    // bool onHost() {return device == -1; }
    // bool onDevice() {return !onHost(); }
    MemoryRef slice(size_t beg, size_t end) {return MemoryRef{offset + beg, end - beg, device, origin}; }
    MemoryRef slice_n(size_t beg, size_t bytes) {return slice(beg, beg + bytes); }

    // template <typename T>
    // operator T*() { return reinterpret_cast<T *>(ptr); }
    // template <typename T>
    // operator const T*() const { return reinterpret_cast<T *>(ptr); }

    uint8_t *ptr() {
      return origin ? static_cast<uint8_t*>(origin->data()) : nullptr;
    }

    template <typename T>
    operator T*() { return reinterpret_cast<T *>(origin->data()); }
    template <typename T>
    operator const T*() const { return reinterpret_cast<T *>(origin->data()); }
  };


  // template <typename T>
  // MemoryRef slice_n(MemoryRef mem, size_t beg, size_t num) {
  //   return mem.slice(std::min(beg * sizeof(T), mem.size), std::min((beg + num) * sizeof(T), mem.size));
  // }

  // template <typename T>
  // MemoryRef slice_n(HostVector<T> &vec, size_t beg, size_t num) {
  //   return slice_n<T>(MemoryRef{vec}, beg, num);
  // }

  // template <typename T>
  // MemoryRef slice(HostVector<T> &vec, size_t beg, size_t end) {
  //   return slice_n(vec, beg, end - beg);
  // }

  /*
   * Host CallBack 
   */
  // template<typename T, void (T::*f)(cudaStream_t, cudaError_t)>
  // void HostCallback(cudaStream_t s, cudaError_t err, void *data) {
  //   std::invoke(f, static_cast<T *>(data), s , err);
  // }

  // template<typename T, void (T::*f)(cudaStream_t, cudaError_t) = &T::operator()>
  // void addHostCallback(cudaStream_t s, T &obj) {
  //   T *obj_ptr = &obj;
  //   void *data = static_cast<void *>(obj_ptr);
  //   auto c = &HostCallback<T, f>;
  //   StreamAddCallback(s, c, data);
  // }

  // template<typename T, void (T::*f)(cudaStream_t, cudaError_t) = &T::operator()>
  // void addHostCallback(cudaStream_t s, T *obj_ptr) {
  //   addHostCallback<T, f>(s, *obj_ptr);
  // }

  // struct CallbackTagWrapper {
  //   using callback_t = std::function<void (int, cudaStream_t, cudaError_t)>;
  //   const int tag;
  //   callback_t func;
  //   CallbackTagWrapper(int _tag, callback_t _func): tag(_tag), func(_func) {};

  //   void operator()(cudaStream_t s, cudaError_t e) {
  //     std::invoke(func, tag, s, e);
  //   }
  // };

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

  extern Platform platform;
}