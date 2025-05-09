#include <vortex-util.h>

#include <iostream>
#include <fstream>
#include <cassert>

#define CUDA_CHECK_ERROR(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(error); \
    } \
}

namespace vortex::cuda {
  /*
   * runtime API Wrappers
   */

  void HostMalloc(void **ptr, size_t size) {
    CUDA_CHECK_ERROR(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
  }

  void HostFree(void *ptr) { CUDA_CHECK_ERROR(cudaFreeHost(ptr)); }

  void DeviceMalloc(void **ptr, size_t size) { CUDA_CHECK_ERROR(cudaMalloc(ptr, size)); }

  void DeviceFree(void *ptr) { CUDA_CHECK_ERROR(cudaFree(ptr)); }
  // void CudaIpcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset, uint64_t size)
  void MemcpyAsync(MemoryRef dst, MemoryRef src, std::shared_ptr<CudaIpcConnection> connection) {
    assert(dst.size == src.size);
    std::cout << "Memcpy L" << std::endl;
    std::cout << dst.offset << " " << src.offset << " " << src.size << " " << std::endl;
    if (src.origin == nullptr || dst.origin == nullptr) {
      std::cout << "???";
    }
    connection->write(*(dst.origin), dst.offset, *(src.origin), src.offset, src.size);
    std::cout << "Memcpy R" << std::endl;
    // if (dst.device == -1 && src.device != -1) {
    //   CUDA_CHECK_ERROR(cudaMemcpyAsync(dst.ptr(), src.ptr(), src.size, cudaMemcpyDeviceToHost, s));
    // } else if (dst.device != -1 && src.device == -1) {
    //   CUDA_CHECK_ERROR(cudaMemcpyAsync(dst.ptr(), src.ptr(), src.size, cudaMemcpyHostToDevice, s));
    // } else if (dst.device != -1 && src.device != -1) {
    //   CUDA_CHECK_ERROR(cudaMemcpyPeerAsync(dst.ptr(), dst.device, src.ptr(), src.device, src.size, s));
    // } else {
    //   assert(0);
    // }
  }

  void MemsetAsync(MemoryRef dst, int value, cudaStream_t s) {
    CUDA_CHECK_ERROR(cudaMemsetAsync(dst.ptr(), value, dst.size, s));
  }

  std::tuple<size_t, size_t> MemGetInfo() {
    size_t left, total;
    CUDA_CHECK_ERROR(cudaMemGetInfo(&left, &total));
    return {left, total};
  }

  void DeviceSynchronize() {
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  }

  int GetDevice() {
    int r;
    CUDA_CHECK_ERROR(cudaGetDevice(&r));
    return r;
  }

  void SetDevice(int d) { CUDA_CHECK_ERROR(cudaSetDevice(d)); }

  int GetDeviceCount() {
    int c;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&c));
    return c;
  }

  void DeviceEnablePeerAccess(int peer) { CUDA_CHECK_ERROR(cudaDeviceEnablePeerAccess(peer, 0)); }

  bool DeviceCanAccessPeer(int device, int peer) {
    int r;
    CUDA_CHECK_ERROR(cudaDeviceCanAccessPeer(&r, device, peer));
    return r;
  }

  cudaDeviceProp GetDeviceProperties(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
    return prop;
  }

  // --- Stream
  cudaStream_t StreamCreate() {
    cudaStream_t r;
    CUDA_CHECK_ERROR(cudaStreamCreate(&r));
    return r;
  }

  void StreamDestroy(cudaStream_t s) { CUDA_CHECK_ERROR(cudaStreamDestroy(s)); }

  bool StreamQuery(cudaStream_t s) {
    cudaError_t err = cudaStreamQuery(s);
    if (err == cudaSuccess)
      return true;
    else if (err == cudaErrorNotReady)
      return false;
    else
      throw std::runtime_error("Failed to query stream");
  }

  void StreamSynchronize(cudaStream_t s) {
    CUDA_CHECK_ERROR(cudaStreamSynchronize(s));
  }

  void StreamWaitEvent(cudaStream_t s, cudaEvent_t e) {
    CUDA_CHECK_ERROR(cudaStreamWaitEvent(s, e, 0));
  }

  void StreamAddCallback(cudaStream_t s, cudaStreamCallback_t callback, void *userData) {
    CUDA_CHECK_ERROR(cudaStreamAddCallback(s, callback, userData, 0));
  }

  // --- Event
  cudaEvent_t EventCreate() {
    cudaEvent_t e;
    CUDA_CHECK_ERROR(cudaEventCreate(&e));
    return e;
  }

  void EventDestroy(cudaEvent_t e) { CUDA_CHECK_ERROR(cudaEventDestroy(e)); }

  void EventRecord(cudaEvent_t e, cudaStream_t s) {
    CUDA_CHECK_ERROR(cudaEventRecord(e, s));
  }

  void EventSynchronize(cudaEvent_t e) { CUDA_CHECK_ERROR(cudaEventSynchronize(e)); }

  bool EventQuery(cudaEvent_t e) {
    cudaError_t err = cudaEventQuery(e);
    if (err == cudaSuccess)
      return true;
    else if (err == cudaErrorNotReady)
      return false;
    else
      throw std::runtime_error("Failed to query Event");
  }

  float EventElapsedTime(cudaEvent_t start, cudaEvent_t stop) {
    float r;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&r, start, stop));
    return r;
  }

  // --- Control Primitives Proxy
  Stream::Stream() : stream_(StreamCreate()), device_(GetDevice()) {}

  Stream::Stream(Stream &&s) {
    if (&s != this) {
      this->stream_ = s.stream_;
      this->device_ = s.device_;
      s.device_ = -1;
    }
  }

  Stream::~Stream() {
    if (device_ != -1) {
      StreamDestroy(stream_);
    }
  }

  Event::Event() : e_(EventCreate()), device_(GetDevice()) {}

  Event::Event(Event &&e) {
    if (&e != this) {
      this->e_ = e.e_;
      this->device_ = e.device_;
      e.device_ = -1;
    }
  }

  Event::~Event() {
    if (device_ != -1) {
      EventDestroy(e_);
    }
  }

  // --- Memory Primitive Proxy
  DeviceGuard::DeviceGuard(int id) : original_device_id(GetDevice()), current_device_id(id) {
    SetDevice(current_device_id);
  }

  DeviceGuard::~DeviceGuard() {
    SetDevice(original_device_id);
  }

  DeviceMemory::DeviceMemory(size_t size) : size_(size), device_(GetDevice()) {
    void *p;
    DeviceMalloc(&p, size_);
    ptr_.reset(p);
  }

  /*
   * Platform Information
   */
  void Platform::EnableAllPeerAccess() {
    for (int i = 0; i < deviceCount_; i++) {
      DeviceGuard on(i);
      for (int j = 0; j < deviceCount_; j++) {
        if (i == j) continue;
        // DeviceEnablePeerAccess(j);
      }
    }
  }

  // void Platform::warmUp() {
  //   std::vector<DeviceMemory> dev;
  //   std::vector<Stream> stream;
  //   for (int d = 0; d < deviceCount_; d++) {
  //     DeviceGuard on(d);
  //     dev.emplace_back(1000'000);
  //     stream.emplace_back();
  //   }
  //   HostVector<uint8_t> host(1000'000);
  //   for (int di = 0; di < deviceCount_; di++) {
  //     for (int dj = 0; dj < deviceCount_; dj++) {
  //       MemcpyAsync(dev[di], dev[dj], stream[di]);
  //     }
  //   }
  //   for (int d = 0; d < deviceCount_; d++) {
  //     MemcpyAsync(dev[d], host, stream[d]);
  //     MemcpyAsync(host, dev[d], stream[d]);
  //   }
  //   for (int d = 0; d < deviceCount_; d++) {
  //     stream[d].synchronize();
  //   }
  // }

  Platform::Platform() : deviceCount_(GetDeviceCount()), deviceProps_(deviceCount_) {
    EnableAllPeerAccess();
    // warmUp();
    for (int i = 0; i < deviceCount_; i++) {
      deviceProps_[i] = GetDeviceProperties(i);
    }
  }

  Platform platform;
}
