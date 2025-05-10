#include <io-sched.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>

namespace vortex::sched {

using vortex::cuda::MemoryRef;
using vortex::cuda::MemcpyAsync;
using mscclpp::Endpoint;
using mscclpp::Transport;

void sleep_for(std::chrono::microseconds v) {
  using namespace std::chrono;
  auto n = high_resolution_clock::now();
  while (n + v > high_resolution_clock::now()) continue;
}

void LoadBalancingSched::_produceTasks(std::vector<IOTask> &tasks, MemoryRef dst, MemoryRef src, size_t gran) {
  assert(dst.size == src.size);

  for (size_t i = 0; i < dst.size; i += gran) {
    size_t beg = i, end = std::min(dst.size, i + gran);
    int id = tasks.size();
    tasks.emplace_back(IOTask{dst.slice(beg, end), src.slice(beg, end), id, false});
  }
}

void LoadBalancingSched::reset(
  const std::vector<MemoryRef> &H2Ddsts, const std::vector<MemoryRef> &H2Dsrcs,
  const std::vector<MemoryRef> &D2Hdsts, const std::vector<MemoryRef> &D2Hsrcs,
  size_t gran
) {
  printf ("::reset\n");
  h2dCnt_ = 0;
  d2hCnt_ = 0;
  h2dIssued_ = 0;
  d2hIssued_ = 0;
  H2Dtraffic_ = 0;
  D2Htraffic_ = 0;

  h2dTasks_.clear();
  d2hTasks_.clear();
  for (size_t i = 0; i < H2Ddsts.size(); i++) {
    H2Dtraffic_ += H2Ddsts[i].size;
    _produceTasks(h2dTasks_, H2Ddsts[i], H2Dsrcs[i], gran);
  }
  for (size_t i = 0; i < D2Hdsts.size(); i++) {
    D2Htraffic_ += D2Hdsts[i].size;
    _produceTasks(d2hTasks_, D2Hdsts[i], D2Hsrcs[i], gran);
  }
  printf ("H2Dtraffic_ = %d\n", (int)H2Dtraffic_);
  printf ("D2Htraffic_ = %d\n", (int)D2Htraffic_);

  std::fill(robh2d_.begin(), robh2d_.end(), 0);
  std::fill(robd2h_.begin(), robd2h_.end(), 0);
  assert(h2dTasks_.size() < 10000);
  assert(d2hTasks_.size() < 10000);

  double readRatio = (H2Dtraffic_ * 1.0) / (H2Dtraffic_ + D2Htraffic_);
  if (readRatio < 0.75) {
    fixedD2H = 1;
  }
}

void LoadBalancingSched::reset(MemoryRef H2Ddst, MemoryRef H2Dsrc, MemoryRef D2Hdst, MemoryRef D2Hsrc, size_t gran) {
  reset(
    std::vector<MemoryRef>{H2Ddst}, std::vector<MemoryRef>{H2Dsrc},
    std::vector<MemoryRef>{D2Hdst}, std::vector<MemoryRef>{D2Hsrc},
    gran
  );
}

IOTask LoadBalancingSched::nextH2D([[maybe_unused]] int id) {
  size_t i = h2dCnt_.fetch_add(1);
  if (i < h2dTasks_.size()) {
    h2dIssued_.fetch_add(h2dTasks_[i].dst.size);
    return h2dTasks_[i];
  } else {
    return IOTask{MemoryRef{}, MemoryRef{}, -1, true};
  }
}

IOTask LoadBalancingSched::nextD2H(int id) {
  if (id > 4 + fixedD2H) { // hyper paramter tuning point
    size_t H2D_equiv = (1.0 * h2dIssued_) / H2Dtraffic_ * D2Htraffic_;
    size_t cur_d2h = d2hIssued_;
    bool pause = cur_d2h > H2D_equiv;
    if (pause) {
      return IOTask{MemoryRef{}, MemoryRef{}, -1, false};
    }
  }

  size_t i = d2hCnt_.fetch_add(1);
  if (i < d2hTasks_.size()) {
    d2hIssued_.fetch_add(d2hTasks_[i].dst.size);
    return d2hTasks_[i];
  } else {
    return IOTask{MemoryRef{}, MemoryRef{}, -1, true};
  }
}

IOTask LoadBalancingSched::next(int id) {
  // printf ("next calling: %d\n", id);
  if (id < NUM_GPU) {
    return nextH2D(id);
  } else {
    return nextD2H(id);
  }
}

void LoadBalancingSched::report(int id, int task) {
  printf ("report: success ID = %d, task = %d\n", id, task);
  if (id < NUM_GPU) {
    robh2d_[task] = 1;
  } else {
    robd2h_[task] = 1;
  }
}

void LoadBalancingSched::h2dSent() {
  for (size_t i = 0; i < h2dTasks_.size(); i++) {
    while (robh2d_[i] != 1) continue;
    size_t end = std::min((i + 1) * gran_, H2Dtraffic_);
    h2dcallback_(end);
  }
}

void LoadBalancingSched::d2hSent() {
  for (size_t i = 0; i < d2hTasks_.size(); i++) {
    while (robd2h_[i] != 1) continue;
    size_t end = std::min((i + 1) * gran_, D2Htraffic_);
    d2hcallback_(end);
  }
}

void IndirectLink::caller() {
  // int old1{0}, old2{0};
  while (true) {
    // fmt::print("[{}] loop begin\n", id_);
    size_t curBufSize = bufDst_.size;
    int curSentId = bufTaskId_;
    if (curBufSize > 0) {
      MemcpyAsync(bufDst_, bufs_[cur_].slice(0, curBufSize), connections_[1]); // empty current buffer
      bufDst_ = MemoryRef{};
      bufTaskId_ = -1;
    }

    auto task = nextFn_(id_);
    // fmt::print("[{}] got task\n", id_);
    size_t nextBufSize = task.src.size;
    if (nextBufSize > 0) {
      MemcpyAsync(bufs_[next_].slice(0, nextBufSize), task.src, connections_[0]);
      bufDst_ = task.dst;
      bufTaskId_ = task.id;
    }
    // fmt::print("[{}] sent s1\n", id_);

    if (curBufSize > 0) {
      // gpuio::hip::addHostCallback<IndirectLink, &IndirectLink::s2Callback>(*streams_[1], this);
    }
    if (nextBufSize > 0) {
      // gpuio::hip::addHostCallback<IndirectLink, &IndirectLink::s1Callback>(*streams_[0], this);
    }
    // fmt::print("[{}] added callback, lastSentId_: {} \n", id_, lastSentId_);

    if (lastSentId_ >= 0) { // report back the finished data
      reportFn_(id_, lastSentId_);
      lastSentId_ = -1;
    }
    lastSentId_ = curSentId;

    // swap buffer pointer
    std::swap(cur_, next_);

    if (curBufSize > 0) {
      // while (old2 == stage2Cnt_) continue; // wait until previous transfer all finish
      connections_[1]->writeSync();
    }
    if (nextBufSize > 0) {
      // while (old1 == stage1Cnt_) continue;
      connections_[0]->writeSync();
    }
    if (!task.done && nextBufSize == 0 && curBufSize == 0) { // scheduler pause the link
      using namespace std::chrono_literals;
      sleep_for(10us);
    }
    // old1 = stage1Cnt_;
    // old2 = stage2Cnt_;

    if (task.done) break; // last transfer, break
  }

  if (lastSentId_ >= 0) { // report back the finished data
    reportFn_(id_, lastSentId_);
    lastSentId_ = -1;
  }

} 

void DirectLink::caller() {
  // int old{0};
  while (true) {
    auto task = nextFn_(id_);
    if (task.dst.size > 0) {
      MemcpyAsync(task.dst, task.src, connection_); // empty current buffer
      // gpuio::hip::addHostCallback<DirectLink, &DirectLink::callback>(*s_, this);
      // fmt::print("[{}] sent s1\n", id_);
      // while (old == cnt_) continue;
      connection_->writeSync();
      if (task.dst.size > 0) {
        reportFn_(id_, task.id);
      }
      // old = cnt_;
    }

    if (task.done) break;
  }
}


ExchangeContextOwning::ExchangeContextOwning(size_t gran, std::shared_ptr<Context> context, const std::vector<Endpoint> &ep): gran_(gran), context_(context) {
  for (int d = 0; d < NUM_GPU; d++) {
    vortex::cuda::DeviceGuard on(d);
    void *buf[4];
    for (int i = 0; i < 4; ++i)
      MSCCLPP_CUDATHROW(cudaMalloc(&buf[i], gran));

    h2dbufs_.emplace_back(0, gran_, d, std::make_shared<RegisteredMemory>(context->registerMemory(buf[0], gran_, Transport::CudaIpc)));
    h2dbufs_.emplace_back(0, gran_, d, std::make_shared<RegisteredMemory>(context->registerMemory(buf[1], gran_, Transport::CudaIpc)));
    d2hbufs_.emplace_back(0, gran_, d, std::make_shared<RegisteredMemory>(context->registerMemory(buf[2], gran_, Transport::CudaIpc)));
    d2hbufs_.emplace_back(0, gran_, d, std::make_shared<RegisteredMemory>(context->registerMemory(buf[3], gran_, Transport::CudaIpc)));

    // h2dstreams_.emplace_back();
    // h2dstreams_.emplace_back();
    // d2hstreams_.emplace_back();
    // d2hstreams_.emplace_back();
    h2dconnections_.push_back(std::dynamic_pointer_cast<CudaIpcConnection>(context->connectWithNewStream(ep[0], ep[d+1])));
    h2dconnections_.push_back(std::dynamic_pointer_cast<CudaIpcConnection>(context->connectWithNewStream(ep[0], ep[d+1])));
    d2hconnections_.push_back(std::dynamic_pointer_cast<CudaIpcConnection>(context->connectWithNewStream(ep[d+1], ep[0])));
    d2hconnections_.push_back(std::dynamic_pointer_cast<CudaIpcConnection>(context->connectWithNewStream(ep[d+1], ep[0])));
  }
}

void ExchangeContextOwning::warmup(MemoryRef host) {
  for (int d = 0; d < NUM_GPU; d++) {
    for (int s = 0; s < 2; s++) {
      auto h2ds = h2dS(d, s);
      auto h2db = h2dbufs(d, s);

      // connection1->write(gpuTmpRegMem, gpuTmpOffset, cpuSrcRegMem, cpuSrcOffset, size);
      // gpuio::hip::MemcpyAsync(host, h2db, h2ds);
      for (int dj = 0; dj < NUM_GPU; dj++) {
        if (dj != d) {
          auto nb = h2dbufs(dj, s);
          // gpuio::hip::MemcpyAsync(nb, h2db, h2ds);
        }
      }
      // gpuio::hip::addHostCallback<ExchangeContextOwning, &ExchangeContextOwning::_warmupfunc>(h2ds, this);

      // h2ds->writeSync();

      auto d2hs = d2hS(d, s);
      auto d2hb = d2hbufs(d, s);
      // gpuio::hip::MemcpyAsync(host, d2hb, d2hs);
      for (int dj = 0; dj < NUM_GPU; dj++) {
        if (dj != d) {
          auto nb = d2hbufs(dj, s);
          // gpuio::hip::MemcpyAsync(nb, d2hb, d2hs);
        }
      }
      // gpuio::hip::addHostCallback<ExchangeContextOwning, &ExchangeContextOwning::_warmupfunc>(d2hs, this);
      d2hs->writeSync();
    }
  }
}



} // namespace 