// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
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
constexpr int MAX_QUEUE_LENGTH = 10000;

using mscclpp::RegisteredMemory;
using mscclpp::CudaIpcConnection;
using mscclpp::Context;
using mscclpp::Endpoint;
using mscclpp::DeviceType;

using vortex::cuda::MemoryRef;

/// Structure representing a single I/O task between source and destination memory regions.
/// 
/// @param dst     Destination memory reference.
/// @param src     Source memory reference.
/// @param id      Task identifier.
/// @param done    Flag indicating task completion.
struct IOTask {
  MemoryRef dst, src;
  int id;
  bool done;
};

using nextFn_t = std::function<IOTask(int)>;
using reportFn_t = std::function<void(int, int)>;
using progressCallback_t = std::function<void(size_t)>;

/// Scheduler that balances H2D and D2H traffic.
///
class LoadBalancingSched {
  std::atomic<size_t> h2dCnt_{0}, d2hCnt_{0}, h2dIssued_{0}, d2hIssued_{0};
  size_t H2Dtraffic_{0}, D2Htraffic_{0}, gran_{0};
  std::array<std::atomic<int>, MAX_QUEUE_LENGTH> robh2d_ = {0};
  std::array<std::atomic<int>, MAX_QUEUE_LENGTH> robd2h_ = {0};

  std::vector<IOTask> h2dTasks_, d2hTasks_;

  std::thread h2dtrack_, d2htrack_;
  
  // Callback methods reserved for debug or log.
  static void _h2dcallbackPrint([[maybe_unused]] size_t finished) {}
  static void _d2hcallbackPrint([[maybe_unused]] size_t finished) {}
  progressCallback_t h2dcallback_ = &LoadBalancingSched::_h2dcallbackPrint; 
  progressCallback_t d2hcallback_ = &LoadBalancingSched::_d2hcallbackPrint;

  /// Produces transfer tasks by dividing the total data range.
  ///
  /// @param tasks   Output vector of IOTasks.
  /// @param dst     Destination memory reference.
  /// @param src     Source memory reference.
  /// @param gran    Size of each transfer task in bytes.
  void _produceTasks(std::vector<IOTask> &tasks, MemoryRef dst, MemoryRef src, size_t gran);

  int fixedD2H = 0;
public:
  LoadBalancingSched() = default;

  /// Resets scheduler tasks.
  void reset(
    const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs,
    const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs,
    size_t gran
  );
  void reset(MemoryRef H2Ddst, MemoryRef H2Dsrc, MemoryRef D2Hdst, MemoryRef D2Hsrc, size_t gran);

  /// Fetches the next H2D I/O task for a given worker ID.
  ///
  /// @param id    Worker ID.
  /// @return      IOTask for the worker.
  IOTask nextH2D(int id); 

  /// Fetches the next D2H I/O task for a given worker ID.
  ///
  /// @param id    Worker ID.
  /// @return      IOTask for the worker.
  IOTask nextD2H(int id); 

  /// Fetches the next I/O task based on the worker ID and direction.
  ///
  /// @param id    Worker ID.
  /// @return      IOTask for the worker.
  IOTask next(int id); 

  /// Reports the completion of a task by a given worker.
  ///
  /// @param id     Worker ID.
  /// @param task   Task ID.
  void report(int id, int task); 


  /// Return the next function.
  operator nextFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::next, this, _1);
  }

  /// Return the report function.
  operator reportFn_t() {
    using namespace std::placeholders;
    return std::bind(&LoadBalancingSched::report, this, _1, _2);
  }

  /// Tracks the progress of H2D transfers.
  void h2dSent();

  /// Tracks the progress of D2H transfers.
  void d2hSent(); 
  
  /// Launches internal tracking threads for H2D and D2H transfers.
  void launch() {
    h2dtrack_ = std::thread(&LoadBalancingSched::h2dSent, this);
    d2htrack_ = std::thread(&LoadBalancingSched::d2hSent, this);
  }

  /// Waits for the tasks to finish.
  void sync() {
    h2dtrack_.join();
    d2htrack_.join();
  }
};

struct IndirectLink {
  int cur_{1}, next_{0};

  MemoryRef dst_, src_; 
  std::array<MemoryRef, 2> bufs_;
  MemoryRef bufDst_;
  int bufTaskId_;
  int lastSentId_;
  std::array<std::shared_ptr<CudaIpcConnection>, 2> connections_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_;

public:
  /// Constructor for IndirectLink class.
  /// Initializes the transfer GPU using two buffers and two connections (CPU - GPU, and GPU - Target GPU).
  IndirectLink(MemoryRef buf1, MemoryRef buf2, std::shared_ptr<CudaIpcConnection> connection1, std::shared_ptr<CudaIpcConnection> connection2,
    nextFn_t nextFn, reportFn_t reportFn, int id)
   : bufs_({buf1, buf2}), connections_({connection1, connection2}), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}

  void reset() {
    cur_ = 1;
    next_ = 0;
    bufDst_ = MemoryRef{};
    bufTaskId_ = -1;
    lastSentId_ = -1; 
  }

  void caller(); 
  void launch() { t_ = std::thread(&IndirectLink::caller, this); }
  void sync() { t_.join(); }
};

struct DirectLink {
  std::shared_ptr<CudaIpcConnection> connection_;
  int lastSentId_;

  std::thread t_;
  nextFn_t nextFn_;
  reportFn_t reportFn_;

  int id_ = -1;
  
public:
  /// Constructor for DirectLink class.
  /// Initializes the target GPU using one connections (CPU - Target GPU).
  DirectLink(std::shared_ptr<CudaIpcConnection> connection, nextFn_t nextFn, reportFn_t reportFn, int id)
    : connection_(connection), nextFn_(nextFn), reportFn_(reportFn), id_(id) {}

  void reset() {
    lastSentId_ = -1;
  }


  void caller(); 
  void launch() { t_ = std::thread(&DirectLink::caller, this); }
  /// Waits for the internal thread to complete.
  void sync() { t_.join(); }
};


/// Utility function to sleep for a given duration in microseconds.
///
/// @param v   Duration to sleep.
void sleep_for(std::chrono::microseconds v); 

/// Manages memory and connections used for data exchange.
///
/// Owns the buffers and CudaIpcConnections for H2D and D2H communication.
class ExchangeContextOwning {
  size_t gran_;
  std::shared_ptr<Context> context_;
  std::vector<MemoryRef> h2dbufs_, d2hbufs_;
  std::vector<std::shared_ptr<CudaIpcConnection>> h2dconnections_, d2hconnections_;

public:
  /// Constructs the context with given endpoints and granularity.
  ///
  /// @param context       Shared context for memory registration.
  /// @param ep            Endpoints including CPU and GPUs.
  /// @param target_gpu    Index of the target GPU.
  /// @param gran          Transfer granularity in bytes.
  ExchangeContextOwning(std::shared_ptr<Context> context, const std::vector<Endpoint> &ep, size_t target_gpu, size_t gran);

  /// Returns H2D buffer at given link and id (direction).
  MemoryRef h2dbufs(int link, int id) { return h2dbufs_[link * 2 + id]; }

  /// Returns D2H buffer at given link and id (direction).
  MemoryRef d2hbufs(int link, int id) { return d2hbufs_[link * 2 + id]; }

  /// Returns H2D connection at given link and id (direction).
  std::shared_ptr<CudaIpcConnection> h2dS(int link, int id) { return h2dconnections_[link * 2 + id]; }

  /// Returns D2H connection at given link and id (direction).
  std::shared_ptr<CudaIpcConnection> d2hS(int link, int id) { return d2hconnections_[link * 2 + id]; }

  /// Returns configured transfer granularity.
  size_t size() const { return gran_; }

  /// Performs warmup
  ///
  /// @param host   Host memory as a buffer.
  void warmup(MemoryRef host); 
};

template <typename Sched>
class Exchange {
  ExchangeContextOwning cxt_;
  Sched sched_;

  std::vector<std::unique_ptr<DirectLink>> dlinks_;
  std::vector<std::unique_ptr<IndirectLink>> ilinks_;
  /// Split a large memory reference into segments based on the shapes in ps
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
  /// Convert memory from RegisteredMemory (MSCCL) format to MemoryRef (Vortex) format
  std::vector<MemoryRef> _convert(const std::vector<RegisteredMemory> &regMemVec) {
    std::vector<MemoryRef> memoryRef;
    for (const auto &regMem: regMemVec) {
      memoryRef.emplace_back(regMem);
    }
    return memoryRef;
  }
  void _launch() {
    sched_.launch();
    for (auto &dl: dlinks_) { dl->launch(); }
    for (auto &il: ilinks_) { il->launch(); }
  }
  void _reset(
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
public:
  /// Constructor for Exchange class.
  /// Initializes data transfer context and links between CPU and GPUs.
  ///
  /// @param context         Shared pointer to the context object for memory and IPC setup.
  /// @param ep              List of endpoints (first is CPU, followed by NUM_GPU GPUs).
  /// @param target_gpu      Target GPU (0 ~ NUM_GPU-1)
  /// @param gran            Transfer granularity in bytes (default: 20MB).

  Exchange(
    std::shared_ptr<Context> context,
    const std::vector<Endpoint> &ep,
    size_t target_gpu,
    size_t gran = 20'000'000
  ) : cxt_(context, ep, target_gpu, gran) {
    if (!context) {
      throw mscclpp::Error("Context is nullptr.", mscclpp::ErrorCode::InvalidUsage);
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

  /// Performs warmup for measurements.
  ///
  /// @param context  Shared pointer to the context
  void warmup(std::shared_ptr<Context> context) {
    if (!context) {
      throw mscclpp::Error("Context is nullptr.", mscclpp::ErrorCode::InvalidUsage);
    }
    uint8_t *mem;
    auto gran = cxt_.size();

    MSCCLPP_CUDATHROW(cudaMallocHost(&mem, gran));
    MemoryRef ref(0, gran, -1, std::make_shared<RegisteredMemory>(context->registerCpuMemory(mem, gran, mscclpp::Transport::CudaIpc)));
    cxt_.warmup(ref);
    MSCCLPP_CUDATHROW(cudaFreeHost(mem));
  }

  /// Launches the exchange using vectors of registered memory regions.
  ///
  /// @param H2Ddsts   Host-to-device destinations (target GPU).
  /// @param H2Dsrcs   Host-to-device sources (CPU).
  /// @param D2Hdsts   Device-to-host destinations (CPU).
  /// @param D2Hsrcs   Device-to-host sources (target GPU).
  void launch(const std::vector<RegisteredMemory> &H2Ddsts, const std::vector<RegisteredMemory> &H2Dsrcs, 
    const std::vector<RegisteredMemory> &D2Hdsts, const std::vector<RegisteredMemory> &D2Hsrcs) {
    auto H2Ddsts_ = _convert(H2Ddsts);
    auto H2Dsrcs_ = _convert(H2Dsrcs);
    auto D2Hdsts_ = _convert(D2Hdsts);
    auto D2Hsrcs_ = _convert(D2Hsrcs);

    _reset(H2Ddsts_, H2Dsrcs_, D2Hdsts_, D2Hsrcs_);
    _launch();
  }

  /// Convenience overload for launching a single H2D and D2H transfer pair.
  ///
  /// @param H2Ddst    Host-to-device destination (target GPU).
  /// @param H2Dsrc    Host-to-device source (CPU).
  /// @param D2Hdst    Device-to-host destination (CPU).
  /// @param D2Hsrc    Device-to-host source (target GPU).
  void launch(const RegisteredMemory &H2Ddst, const RegisteredMemory &H2Dsrc, 
    const RegisteredMemory &D2Hdst, const RegisteredMemory &D2Hsrc) {
    launch(
      std::vector<RegisteredMemory>{H2Ddst},
      std::vector<RegisteredMemory>{H2Dsrc},
      std::vector<RegisteredMemory>{D2Hdst},
      std::vector<RegisteredMemory>{D2Hsrc}
    );
  }

  /// Blocks until all data transfers are complete.
  void sync() {
    sched_.sync();
    for (auto &dl: dlinks_) { dl->sync(); }
    for (auto &il: ilinks_) { il->sync(); }
  }
};

using LoadBalancedExchange = Exchange<LoadBalancingSched>;

}