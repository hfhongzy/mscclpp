#pragma once

#include <vortex-util.h>

#include <array>
#include <atomic>
#include <cassert>
#include <thread>

#include "registered_memory.hpp"
#include "connection.hpp"
#include "context.hpp"
#include "endpoint.hpp"
#include "mscclpp/core.hpp"

namespace vortex::sched {

constexpr int NUM_GPU = 4;

using mscclpp::RegisteredMemory;
using mscclpp::CudaIpcConnection;
using mscclpp::Context;
using mscclpp::Endpoint;
using mscclpp::DeviceType;

using vortex::cuda::MemoryRef;
// using vortex::cuda::Stream;
// using vortex::cuda::DeviceMemory;

// using gpuio::cuda::MemcpyAsync;
// using gpuio::cuda::HostVector;

struct IOTask {
  MemoryRef dst, src;
  int id;
  bool done;
};

using nextFn_t = std::function<IOTask(int)>;
using reportFn_t = std::function<void(int, int)>;
using progressCallback_t = std::function<void(size_t)>;

class LoadBalancingSched {
  std::atomic<size_t> h2dCnt_{0}, d2hCnt_{0}, h2dIssued_{0}, d2hIssued_{0};
  size_t H2Dtraffic_{0}, D2Htraffic_{0}, gran_{0};
  std::array<std::atomic<int>, 10000> robh2d_ = {0};
  std::array<std::atomic<int>, 10000> robd2h_ = {0};

  std::vector<IOTask> h2dTasks_, d2hTasks_;

  std::thread h2dtrack_, d2htrack_;

  static void _h2dcallbackPrint(size_t finished) {}
  static void _d2hcallbackPrint(size_t finished) {}
  progressCallback_t h2dcallback_ = &LoadBalancingSched::_h2dcallbackPrint; 
  progressCallback_t d2hcallback_ = &LoadBalancingSched::_d2hcallbackPrint;

  void _produceTasks(std::vector<IOTask> &tasks, MemoryRef dst, MemoryRef src, size_t gran);

  int fixedD2H = 0;
public:
  LoadBalancingSched() = default;
  
  void reset(progressCallback_t h2dcallback, progressCallback_t d2hcallback) {
    h2dcallback_ = h2dcallback;
    d2hcallback_ = d2hcallback;
  }

  void reset(
    const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs,
    const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs,
    size_t gran
  );

  void reset(MemoryRef H2Ddst, MemoryRef H2Dsrc, MemoryRef D2Hdst, MemoryRef D2Hsrc, size_t gran);

  IOTask nextH2D(int); 
  IOTask nextD2H(int id); 
  IOTask next(int id); 
  void report(int id, int task); 

  operator nextFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::next, this, _1);
  }

  operator reportFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::report, this, _1, _2);
  }

  void h2dSent(); 
  void d2hSent(); 

  void launch() {
    h2dtrack_ = std::thread(&LoadBalancingSched::h2dSent, this);
    d2htrack_ = std::thread(&LoadBalancingSched::d2hSent, this);
  }

  void sync() {
    h2dtrack_.join();
    d2htrack_.join();
  }
};

struct IndirectLink {
  std::atomic<int> stage1Cnt_{0}, stage2Cnt_{0};
  int cur_{1}, next_{0};

  MemoryRef dst_, src_; 
  std::array<MemoryRef, 2> bufs_;
  MemoryRef bufDst_;
  int bufTaskId_;
  int lastSentId_;
  // std::array<Stream*, 2> streams_;
  std::array<std::shared_ptr<CudaIpcConnection>, 2> connections_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_;

public:
  IndirectLink(MemoryRef buf1, MemoryRef buf2, std::shared_ptr<CudaIpcConnection> connection1, std::shared_ptr<CudaIpcConnection> connection2,
    nextFn_t nextFn, reportFn_t reportFn, int id)
   : bufs_({buf1, buf2}), connections_({connection1, connection2}), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}

  void reset() {
    stage1Cnt_ = 0;
    stage2Cnt_ = 0;
    cur_ = 1;
    next_ = 0;
    bufDst_ = MemoryRef{};
    bufTaskId_ = -1;
    lastSentId_ = -1; 
  }

  void s1Callback(cudaStream_t, cudaError_t) { stage1Cnt_.fetch_add(1); }
  void s2Callback(cudaStream_t, cudaError_t) { stage2Cnt_.fetch_add(1); }
  void caller(); 
  void launch() { t_ = std::thread(&IndirectLink::caller, this); }
  void sync() { t_.join(); }
};

struct DirectLink {
  std::atomic<int> cnt_{0};
  // Stream *s_ = nullptr;
  std::shared_ptr<CudaIpcConnection> connection_;
  int lastSentId_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_ = -1;

  
public:
  DirectLink(std::shared_ptr<CudaIpcConnection> connection, nextFn_t nextFn, reportFn_t reportFn, int id)
    : connection_(connection), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}

  void reset() {
    cnt_ = 0;
    lastSentId_ = -1;
  }

  void callback(cudaStream_t, cudaError_t) { cnt_.fetch_add(1); }
  void caller(); 
  void launch() { t_ = std::thread(&DirectLink::caller, this); }
  void sync() { t_.join(); }
};


void sleep_for(std::chrono::microseconds v); 

class ExchangeContextOwning {
  size_t gran_;
  std::shared_ptr<Context> context_;
  std::vector<MemoryRef> h2dbufs_, d2hbufs_;
  // std::vector<Stream> h2dstreams_, d2hstreams_;
  std::vector<std::shared_ptr<CudaIpcConnection>> h2dconnections_, d2hconnections_;

public:
  void _warmupfunc(cudaStream_t, cudaError_t) {}
  ExchangeContextOwning(size_t gran, std::shared_ptr<Context> context, const std::vector<Endpoint> &ep);

  MemoryRef h2dbufs(int link, int id) { return h2dbufs_[link * 2 + id]; }
  MemoryRef d2hbufs(int link, int id) { return d2hbufs_[link * 2 + id]; }
  // Stream &h2dS(int link, int id) { return h2dstreams_[link * 2 + id]; }
  // Stream &d2hS(int link, int id) { return d2hstreams_[link * 2 + id]; }
  std::shared_ptr<CudaIpcConnection> h2dS(int link, int id) { return h2dconnections_[link * 2 + id]; }
  std::shared_ptr<CudaIpcConnection> d2hS(int link, int id) { return d2hconnections_[link * 2 + id]; }
  size_t size() const { return gran_; }
  void warmup(MemoryRef host); 
};

template <typename Sched>
class Exchange {
  ExchangeContextOwning cxt_;
  Sched sched_;

  std::vector<std::unique_ptr<DirectLink>> dlinks_;
  std::vector<std::unique_ptr<IndirectLink>> ilinks_;

  std::vector<MemoryRef> _divideRef(MemoryRef big, const std::vector<MemoryRef> &ps) {
    std::vector<MemoryRef> r;
    size_t cur = 0;
    for (auto p: ps) {
      r.push_back(big.slice(cur, cur + p.size));
      cur += p.size;
    }
    assert(cur == big.size);
    return r;
  }
  
public:
  Exchange(size_t gran, size_t num_gpu, std::shared_ptr<Context> context, const std::vector<Endpoint> &ep) : cxt_(gran, context, ep) {
    if (!context) {
      throw mscclpp::Error("Context is nullptr.", mscclpp::ErrorCode::InvalidUsage);
    }
    if (num_gpu != NUM_GPU) {
      throw mscclpp::Error("The number of GPUs is not consistent with hard-coded parameter NUM_GPU.", mscclpp::ErrorCode::InvalidUsage);
    }
    if (ep.size() != NUM_GPU + 1) {
      throw mscclpp::Error("Invalid endpoints length.", mscclpp::ErrorCode::InvalidUsage);
    }
    if (ep[0].getDeviceType() != DeviceType::CPU) {
      throw mscclpp::Error("The first endpoint is not CPU.", mscclpp::ErrorCode::InvalidUsage);
    }
    for (size_t i = 1; i <= NUM_GPU; ++i) {
      if (ep[i].getDeviceType() != DeviceType::GPU) {
        throw mscclpp::Error("The CPU can only be the first endpoint.", mscclpp::ErrorCode::InvalidUsage);
      }
    }

    dlinks_.emplace_back(std::make_unique<DirectLink>(cxt_.h2dS(0, 0), sched_, sched_, 0));
    dlinks_.emplace_back(std::make_unique<DirectLink>(cxt_.d2hS(0, 0), sched_, sched_, NUM_GPU + 0));

    for (size_t i = 1; i < NUM_GPU; ++i) {
      ilinks_.emplace_back(std::make_unique<IndirectLink>(cxt_.h2dbufs(i, 0), cxt_.h2dbufs(i, 1), cxt_.h2dS(i, 0), cxt_.h2dS(i, 1), sched_, sched_, i));
    }
    for (size_t i = 1; i < NUM_GPU; ++i) {
      ilinks_.emplace_back(std::make_unique<IndirectLink>(cxt_.d2hbufs(i, 0), cxt_.d2hbufs(i, 1), cxt_.d2hS(i, 0), cxt_.d2hS(i, 1), sched_, sched_, NUM_GPU + i));
    }
  }

  void warmup() {
    // HostVector<uint8_t> htmp_(cxt_.size());
    // cxt_.warmup(htmp_);
  }
  
  void reset(MemoryRef dstDevice, MemoryRef srcHost, MemoryRef dstHost, MemoryRef srcDevice) {
    sched_.reset(dstDevice, srcHost, dstHost, srcDevice, cxt_.size());
    for (auto &dl: dlinks_) { dl.reset(); }
    for (auto &il: ilinks_) { il.reset(); }
  }

  void reset(
    const std::vector<MemoryRef> &H2Ddsts, 
    const std::vector<MemoryRef> &H2Dsrcs, 
    const std::vector<MemoryRef> &D2Hdsts, 
    const std::vector<MemoryRef> &D2Hsrcs 
  ) {
    const std::vector<MemoryRef> &H2Ddsts_ = (H2Ddsts.size() == 1 && H2Dsrcs.size() > 1) ? _divideRef(H2Ddsts[0], H2Dsrcs) : H2Ddsts;
    const std::vector<MemoryRef> &D2Hsrcs_ = (D2Hsrcs.size() == 1 && D2Hdsts.size() > 1) ? _divideRef(D2Hsrcs[0], D2Hdsts) : D2Hsrcs;
    sched_.reset(H2Ddsts_, H2Dsrcs, D2Hdsts, D2Hsrcs_, cxt_.size());
    for (auto &dl: dlinks_) { dl->reset(); }
    for (auto &il: ilinks_) { il->reset(); }
  }

  void launch() {
    sched_.launch();
    for (auto &dl: dlinks_) { dl->launch(); }
    for (auto &il: ilinks_) { il->launch(); }
  }
  std::vector<MemoryRef> _convert(const std::vector<RegisteredMemory> &regMemVec) {
    std::vector<MemoryRef> memoryRef;
    for (const auto &regMem: regMemVec) {
      memoryRef.emplace_back(regMem);
    }
    return memoryRef;
  }
  // gpuDstRegMem, cpuSrcRegMem, cpuDstRegMem, gpuSrcRegMem
  void launch(const std::vector<RegisteredMemory> &H2Ddsts, const std::vector<RegisteredMemory> &H2Dsrcs, 
    const std::vector<RegisteredMemory> &D2Hdsts, const std::vector<RegisteredMemory> &D2Hsrcs) {
    auto H2Ddsts_ = _convert(H2Ddsts);
    auto H2Dsrcs_ = _convert(H2Dsrcs);
    auto D2Hdsts_ = _convert(D2Hdsts);
    auto D2Hsrcs_ = _convert(D2Hsrcs);

    reset(H2Ddsts_, H2Dsrcs_, D2Hdsts_, D2Hsrcs_);
    launch();
  }
  void launch(const RegisteredMemory &H2Ddsts, const RegisteredMemory &H2Dsrcs, 
    const RegisteredMemory &D2Hdsts, RegisteredMemory &D2Hsrcs) {
    launch(
      std::vector<RegisteredMemory>{H2Ddsts},
      std::vector<RegisteredMemory>{H2Dsrcs},
      std::vector<RegisteredMemory>{D2Hdsts},
      std::vector<RegisteredMemory>{D2Hsrcs}
    );
  }

  void sync() {
    std::cout << "1::\n" << std::endl;
    sched_.sync();
    std::cout << "2::\n" << std::endl;
    for (auto &dl: dlinks_) { dl->sync(); }
    std::cout << "3::\n" << std::endl;
    for (auto &il: ilinks_) { il->sync(); }
    std::cout << "4::\n" << std::endl;
  }
};

using LoadBalancedExchange = Exchange<LoadBalancingSched>;

} // namespace gpuio::sched::dyn

// namespace gpuio::sched::naive {

// struct NaiveExchange {
//   std::vector<MemoryRef> _divideRef(MemoryRef big, const std::vector<MemoryRef> &ps) {
//     std::vector<MemoryRef> r;
//     size_t cur = 0;
//     for (auto p: ps) {
//       r.push_back(big.slice(cur, cur + p.size));
//       cur += p.size;
//     }
//     assert(cur == big.size);
//     return r;
//   }

//   std::vector<Stream> ss_;

//   NaiveExchange() {}

//   void launch(const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs, 
//     const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs 
//   ) {
//     if (!(H2Ddsts.size() || D2Hsrcs.size())) return;
//     // assert(H2Ddsts.size() || D2Hsrcs.size());
//     int device = H2Ddsts.size() ? H2Ddsts[0].device : D2Hsrcs[0].device;
//     gpuio::cuda::DeviceGuard on(device);
//     ss_.clear();
//     ss_.emplace_back();
//     ss_.emplace_back();
//     // auto &s_ = ss_[0];
//     // fmt::print("{} {} {} {}\n", H2Ddsts.size(), H2Dsrcs.size(), D2Hdsts.size(), D2Hsrcs.size());

//     const std::vector<MemoryRef> &H2Ddsts_ = (H2Ddsts.size() == 1 && H2Dsrcs.size() > 1) ? _divideRef(H2Ddsts[0], H2Dsrcs) : H2Ddsts;
//     const std::vector<MemoryRef> &D2Hsrcs_ = (D2Hsrcs.size() == 1 && D2Hdsts.size() > 1) ? _divideRef(D2Hsrcs[0], D2Hdsts) : D2Hsrcs;

//     // fmt::print("{} {} {} {}\n", H2Ddsts_.size(), H2Dsrcs.size(), D2Hdsts.size(), D2Hsrcs_.size());
//     size_t it = std::max(H2Dsrcs.size(), D2Hdsts.size());
//     for (size_t i = 0; i < it; i++) {
//       if (i < H2Dsrcs.size()) {
//         gpuio::cuda::MemcpyAsync(H2Ddsts_[i], H2Dsrcs[i], ss_[0]);
//       }
//       if (i < D2Hdsts.size()) {
//         gpuio::cuda::MemcpyAsync(D2Hdsts[i], D2Hsrcs_[i], ss_[1]);
//       }
//     }
//     // fmt::print("done\n");
//   }

//   void sync() {
//     if (ss_.size() == 0) return;
//     ss_[0].synchronize();
//     ss_[1].synchronize();
//   }
// };


// } // namespace gpuio::sched::naive