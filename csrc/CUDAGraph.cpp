#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Functions.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#if __has_include(<ATen/cuda/MemPool.h>)
#include <ATen/cuda/MemPool.h>
#define FOUNDRY_HAS_ATEN_CUDA_MEMPOOL 1
#endif
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/driver_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include "CUDAGraph.h"
#include "CUDAGraphInternal.h"
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <cuda.h>
#include <fstream>
#include <boost/json.hpp>
#include <iomanip>
#include <sstream>
#include <vector>
#include "hook.h"
#include "BinaryGraphFormat.h"
#include "GraphDependencies.h"
#include "ExecShare.h"

#include <torch/version.h>

// torch >= 2.13 replaced the per-generator capture RNG state (capturing_,
// offset_intragraph_, capture_prologue/epilogue) with per-capture-id
// CUDAGeneratorCaptureState objects. Both APIs are unexported, so foundry
// carries local copies of whichever one the building torch uses.
#define FOUNDRY_TORCH_PER_CAPTURE_RNG \
  (TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13))

// torch >= 2.14 creates the per-capture state lazily: get_capture_state(id,
// create_if_not_found=true) replaces init_capture_state, and
// CUDAGeneratorImpl::register_graph is gone. libtorch's lazy path registers the
// state with the at::cuda::CUDAGraph it finds through get_graph_from_capture_id,
// which never sees a foundry::CUDAGraph, so foundry creates the capture state of
// every generator it registers up front; libtorch's lookup then finds it.
#define FOUNDRY_TORCH_LAZY_CAPTURE_RNG \
  (TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 14))

// #define FOUNDRY_DEBUG  // Verbose logging (see README "Debugging"); same flag as hook.cpp

namespace at {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG

// Local copies of torch 2.14's unexported CUDAGeneratorCaptureState /
// CUDAGeneratorState members (aten/src/ATen/cuda/CUDAGeneratorImpl.cpp,
// v2.14.1). libtorch's own philox_cuda_state / increase operate on the same
// member data, so these must stay semantically identical to the upstream source.

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::initialize(uint64_t seed) {
  if (is_initialized()) {
    return;
  }

  auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  c10::InferenceMode inference_guard(false);

  // Allocate on the default stream so that the caching allocator routes
  // these tensors to the default memory pool, not the graph's capture pool.
  c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(cudaStreamCaptureModeRelaxed);
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getDefaultCUDAStream());

  rng_state_seed_extragraph_ = at::empty({1}, options);
  rng_state_offset_extragraph_ = at::empty({1}, options);
  // Captured graphs bake in these buffers' addresses: make the storage
  // non-resizable so nothing can reallocate it.
  rng_state_seed_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);
  rng_state_offset_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);

  c10::cuda::getDefaultCUDAStream().synchronize();

  offset_intragraph_ = 0;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::increase(uint64_t increment) {
  TORCH_INTERNAL_ASSERT(offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(offset_intragraph_ <= std::numeric_limits<uint64_t>::max() - increment,
                        "Increment causes overflow in the offset value.");
  offset_intragraph_ += increment;
}

__attribute__((visibility("hidden"))) uint64_t CUDAGeneratorCaptureState::finalize() {
  uint64_t result = offset_intragraph_;
  offset_intragraph_ = 0;
  return result;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::setup_for_replay(
    uint64_t seed, uint64_t philox_offset) {
  TORCH_INTERNAL_ASSERT(is_initialized(), "Capture state not initialized");
  rng_state_seed_extragraph_.fill_(static_cast<int64_t>(seed));
  rng_state_offset_extragraph_.fill_(static_cast<int64_t>(philox_offset));

  // Upstream Note [RNG state tensor lifetime and recordStream]: the replay
  // stream reads these tensors, so a free must wait for it.
  auto stream = c10::cuda::getCurrentCUDAStream();
  c10::cuda::CUDACachingAllocator::recordStream(rng_state_seed_extragraph_.storage().data_ptr(),
                                                stream);
  c10::cuda::CUDACachingAllocator::recordStream(rng_state_offset_extragraph_.storage().data_ptr(),
                                                stream);
}

// Upstream additionally looks up the capturing at::cuda::CUDAGraph and calls
// its register_generator_state when it creates the state. A foundry::CUDAGraph
// is not in that lookup; foundry's callers register the state themselves
// (CUDAGraph::register_generator_state) before creating its capture state.
__attribute__((visibility("hidden"))) CUDAGeneratorCaptureState*
CUDAGeneratorState::get_capture_state(CaptureId_t capture_id, bool create_if_not_found) {
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    if (it != capture_states_.end()) {
      return it->second.get();
    }
    if (!create_if_not_found) {
      return nullptr;
    }
  }

  auto capture_state = make_intrusive<CUDAGeneratorCaptureState>();
  capture_state->initialize(seed_);

  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  auto it = capture_states_.find(capture_id);
  if (it != capture_states_.end()) {
    return it->second.get();
  }
  auto* ptr = capture_state.get();
  capture_states_[capture_id] = std::move(capture_state);
  return ptr;
}

__attribute__((visibility("hidden"))) uint64_t
CUDAGeneratorState::capture_epilogue(CaptureId_t capture_id) {
  auto* capture_state = get_capture_state(capture_id, false);
  if (capture_state) {
    return capture_state->finalize();
  }
  return 0;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::remove_capture_state(
    CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  capture_states_.erase(capture_id);
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::replay_prologue(
    CaptureId_t capture_id, uint64_t wholegraph_increment) {
  if (wholegraph_increment == 0) {
    return;
  }

  auto* capture_state = get_capture_state(capture_id);
  TORCH_INTERNAL_ASSERT(capture_state != nullptr,
                        "replay_prologue called but no capture state found for this capture_id");
  capture_state->setup_for_replay(seed_, philox_offset_per_thread_);
  philox_offset_per_thread_ += wholegraph_increment;
}

#elif FOUNDRY_TORCH_PER_CAPTURE_RNG

// Local copies of torch 2.13's unexported CUDAGeneratorCaptureState /
// CUDAGeneratorState members (aten/src/ATen/cuda/CUDAGeneratorImpl.cpp).
// libtorch's own philox_cuda_state / increase operate on the same member
// data, so these must stay semantically identical to the upstream source.

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::initialize(uint64_t seed) {
  if (is_initialized()) {
    return;
  }

  auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  c10::InferenceMode inference_guard(false);

  // Allocate on the default stream so that the caching allocator routes
  // these tensors to the default memory pool, not the graph's capture pool.
  c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(cudaStreamCaptureModeRelaxed);
  c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getDefaultCUDAStream());

  rng_state_seed_extragraph_ = at::empty({1}, options);
  rng_state_offset_extragraph_ = at::empty({1}, options);

  c10::cuda::getDefaultCUDAStream().synchronize();

  offset_intragraph_ = 0;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::increase(uint64_t increment) {
  TORCH_INTERNAL_ASSERT(offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(offset_intragraph_ <= std::numeric_limits<uint64_t>::max() - increment,
                        "Increment causes overflow in the offset value.");
  offset_intragraph_ += increment;
}

__attribute__((visibility("hidden"))) uint64_t CUDAGeneratorCaptureState::finalize() {
  uint64_t result = offset_intragraph_;
  offset_intragraph_ = 0;
  return result;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorCaptureState::setup_for_replay(
    uint64_t seed, uint64_t philox_offset) {
  TORCH_INTERNAL_ASSERT(is_initialized(), "Capture state not initialized");
  rng_state_seed_extragraph_.fill_(static_cast<int64_t>(seed));
  rng_state_offset_extragraph_.fill_(static_cast<int64_t>(philox_offset));
}

__attribute__((visibility("hidden"))) CUDAGeneratorCaptureState*
CUDAGeneratorState::get_capture_state(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  auto it = capture_states_.find(capture_id);
  if (it != capture_states_.end()) {
    return it->second.get();
  }
  return nullptr;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::init_capture_state(
    CaptureId_t capture_id) {
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    if (capture_states_.count(capture_id)) {
      return;
    }
  }

  auto capture_state = make_intrusive<CUDAGeneratorCaptureState>();
  capture_state->initialize(seed_);

  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  if (!capture_states_.count(capture_id)) {
    capture_states_[capture_id] = std::move(capture_state);
  }
}

__attribute__((visibility("hidden"))) uint64_t
CUDAGeneratorState::capture_epilogue(CaptureId_t capture_id) {
  auto* capture_state = get_capture_state(capture_id);
  if (capture_state) {
    return capture_state->finalize();
  }
  return 0;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::remove_capture_state(
    CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  capture_states_.erase(capture_id);
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::replay_prologue(
    CaptureId_t capture_id, uint64_t wholegraph_increment) {
  if (wholegraph_increment == 0) {
    return;
  }

  auto* capture_state = get_capture_state(capture_id);
  TORCH_INTERNAL_ASSERT(capture_state != nullptr,
                        "replay_prologue called but no capture state found for this capture_id");
  capture_state->setup_for_replay(seed_, philox_offset_per_thread_);
  philox_offset_per_thread_ += wholegraph_increment;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorImpl::register_graph(
    cuda::CUDAGraph* graph) {
  // Upstream forwards to at::cuda::CUDAGraph::register_generator_state; the
  // graph here is really a foundry::CUDAGraph, so route to foundry's method.
  auto foundry_graph = reinterpret_cast<foundry::CUDAGraph*>(graph);
  foundry_graph->register_generator_state(state_);
}

#else  // torch < 2.13: single per-state capture RNG (upstream 2.11 source)

__attribute__((visibility("hidden"))) void CUDAGeneratorState::register_graph(
    cuda::CUDAGraph* graph) {
  at::cuda::assertNotCapturing("Cannot register the state during capturing stage.");

  if (registered_graphs_.empty()) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::unregister_graph(
    cuda::CUDAGraph* graph) {
  TORCH_CHECK(registered_graphs_.find(graph) != registered_graphs_.end(),
              "The graph should be registered to the state");

  registered_graphs_.erase(graph);

  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::increase(uint64_t increment) {
  increment = ((increment + 3) / 4) * 4;
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    TORCH_CHECK(capturing_, "Attempt to increase offset for a CUDA generator not in capture mode.");
    TORCH_INTERNAL_ASSERT(offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    TORCH_INTERNAL_ASSERT(offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
                          "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    TORCH_CHECK(!capturing_, "Offset increment outside graph capture encountered unexpectedly.");
    TORCH_INTERNAL_ASSERT(philox_offset_per_thread_ % 4 == 0,
                          "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::capture_prologue() {
  capturing_ = true;
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(0));
}

__attribute__((visibility("hidden"))) uint64_t CUDAGeneratorState::capture_epilogue() {
  capturing_ = false;
  return offset_intragraph_;
}

__attribute__((visibility("hidden"))) void CUDAGeneratorState::replay_prologue(
    uint64_t wholegraph_increment) {
  at::cuda::assertNotCapturing("Cannot prepare for replay during capturing stage.");
  if (wholegraph_increment) {
    seed_extragraph_.fill_(int64_t(seed_));
    offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
    increase(wholegraph_increment);
  }
}

__attribute__((visibility("hidden"))) void CUDAGeneratorImpl::register_graph(
    cuda::CUDAGraph* graph) {
  auto foundry_graph = reinterpret_cast<foundry::CUDAGraph*>(graph);
  foundry_graph->register_generator_state(state_);
  state_->register_graph(graph);
}

__attribute__((visibility("hidden"))) void CUDAGeneratorImpl::unregister_graph(
    cuda::CUDAGraph* graph) {
  state_->unregister_graph(graph);
}

#endif  // FOUNDRY_TORCH_PER_CAPTURE_RNG

#pragma GCC diagnostic pop
}  // namespace at

namespace foundry {

namespace {

#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
// torch 2.14 dropped CUDAGeneratorImpl::register_graph, the member through which
// foundry reached a generator's private state_. An explicit instantiation is
// exempt from access checking ([temp.spec.general]), which yields a
// pointer-to-member for it; the layout is torch's own, nothing is assumed.
struct GeneratorStateMember {
  using type = c10::intrusive_ptr<at::CUDAGeneratorState> at::CUDAGeneratorImpl::*;
};
template <typename Tag, typename Tag::type Member>
struct PrivateMemberAccess {
  friend typename Tag::type member_pointer(Tag) { return Member; }
};
GeneratorStateMember::type member_pointer(GeneratorStateMember);
template struct PrivateMemberAccess<GeneratorStateMember, &at::CUDAGeneratorImpl::state_>;

c10::intrusive_ptr<at::CUDAGeneratorState> generator_state_of(at::CUDAGeneratorImpl* gen) {
  return gen->*member_pointer(GeneratorStateMember{});
}
#endif  // FOUNDRY_TORCH_LAZY_CAPTURE_RNG

MempoolId_t torch_graph_pool_handle(bool is_user_created = true) {
#if FOUNDRY_HAS_ATEN_CUDA_MEMPOOL
  return at::cuda::MemPool::graph_pool_handle(is_user_created);
#else
  return c10::cuda::MemPool::graph_pool_handle(is_user_created);
#endif
}

}  // namespace

static bool _cuda_graphs_debug = false;

static CUDAGeneratorStateRegistry global_generator_state_registry;

uint64_t CUDAGeneratorStateRegistry::query_state_id(at::CUDAGeneratorState* state) {
  uint64_t result_id = 0;

  auto visit_fn = [&](const auto& value) { result_id = value.second; };

  if (!id_map_.cvisit(state, visit_fn)) {
    result_id = id_counter.fetch_add(1, std::memory_order_relaxed);
    id_map_.emplace(state, result_id);
  }

  return result_id;
}

c10::intrusive_ptr<at::CUDAGeneratorState> CUDAGeneratorStateRegistry::get_state_from_id(
    uint64_t id, uint64_t seed) {
  c10::intrusive_ptr<at::CUDAGeneratorState> result_state;

  auto visit_fn = [&](const auto& value) { result_state = value.second; };

  if (!state_pool_.cvisit(id, visit_fn)) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
    result_state = c10::make_intrusive<at::CUDAGeneratorState>(seed, 0);
#else
    result_state = c10::make_intrusive<at::CUDAGeneratorState>(seed, 0, 0);
#endif
    state_pool_.emplace(id, result_state);
  }

  return result_state;
}

MempoolId_t graph_pool_handle() {
  return torch_graph_pool_handle();
}

void preallocate_cublas_workspaces() {
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublas_handle != nullptr, "Failed to get cuBLAS handle");

  cublasLtHandle_t cublaslt_handle = at::cuda::getCurrentCUDABlasLtHandle();
  TORCH_CHECK(cublaslt_handle != nullptr, "Failed to get cuBLASLt handle");

  void* cublaslt_workspace = at::cuda::getCUDABlasLtWorkspace();
  TORCH_CHECK(cublaslt_workspace != nullptr, "Failed to allocate cuBLASLt workspace");
}

CUDAGraph::CUDAGraph(bool keep_graph)
    : capture_stream_(at::cuda::getCurrentCUDAStream()), keep_graph_(keep_graph) {}

void CUDAGraph::register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  global_generator_state_registry.query_state_id(state.get());
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<at::CUDAGeneratorImpl> cuda_gen =
      c10::dynamic_intrusive_pointer_cast<at::CUDAGeneratorImpl>(generator.getIntrusivePtr());
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
  register_generator_state(generator_state_of(cuda_gen.get()));
#else
  cuda_gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
#endif
}

#if FOUNDRY_TORCH_PER_CAPTURE_RNG
namespace {

// Loaded graphs never went through cudaStreamBeginCapture, so they have no
// driver-issued capture id. Hand each one a synthetic id (high bit set to
// stay clear of driver ids) so the per-capture RNG state map has a key for
// replay_prologue / remove_capture_state.
at::CaptureId_t next_loaded_graph_capture_id() {
  static std::atomic<at::CaptureId_t> counter{1};
  return (static_cast<at::CaptureId_t>(1) << 63) | counter.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace
#endif  // FOUNDRY_TORCH_PER_CAPTURE_RNG

void CUDAGraph::register_generator_state(c10::intrusive_ptr<at::CUDAGeneratorState> state,
                                         uint64_t wholegraph_increment) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  if (capture_id_ == static_cast<CaptureId_t>(-1)) {
    capture_id_ = next_loaded_graph_capture_id();
  }
  // Allocates this (state, capture_id) pair's seed/offset extragraph tensors
  // (the lazy allocation register_graph used to do on the state itself).
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
  state->get_capture_state(capture_id_, /*create_if_not_found=*/true);
#else
  state->init_capture_state(capture_id_);
#endif
#else
  state->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
#endif
  captured_generator_states_[state] = wholegraph_increment;
}

std::pair<uintptr_t, uintptr_t> CUDAGraph::rng_state_addresses(
    const c10::intrusive_ptr<at::CUDAGeneratorState>& state) const {
  auto addresses = [](const at::TensorBase& seed, const at::TensorBase& offset) {
    return std::pair<uintptr_t, uintptr_t>{
        seed.defined() ? reinterpret_cast<uintptr_t>(seed.data_ptr()) : 0,
        offset.defined() ? reinterpret_cast<uintptr_t>(offset.data_ptr()) : 0};
  };
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
  auto* cs = state->get_capture_state(capture_id_, /*create_if_not_found=*/false);
#elif FOUNDRY_TORCH_PER_CAPTURE_RNG
  auto* cs = state->get_capture_state(capture_id_);
#endif
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  if (cs == nullptr)
    return {0, 0};
  return addresses(cs->rng_state_seed_extragraph_, cs->rng_state_offset_extragraph_);
#else
  return addresses(state->seed_extragraph_, state->offset_extragraph_);
#endif
}

bool CUDAGraph::rebind_rng_state(const c10::intrusive_ptr<at::CUDAGeneratorState>& state,
                                 uintptr_t seed_ptr, uintptr_t offset_ptr) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
  auto* cs = state->get_capture_state(capture_id_, /*create_if_not_found=*/false);
#else
  auto* cs = state->get_capture_state(capture_id_);
#endif
  if (cs == nullptr)
    return false;
  auto options = at::TensorOptions().device(at::kCUDA, capture_dev_).dtype(at::kLong);
  cs->rng_state_seed_extragraph_ = at::from_blob(reinterpret_cast<void*>(seed_ptr), {1}, options);
  cs->rng_state_offset_extragraph_ =
      at::from_blob(reinterpret_cast<void*>(offset_ptr), {1}, options);
  cs->rng_state_seed_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);
  cs->rng_state_offset_extragraph_.storage().unsafeGetStorageImpl()->set_resizable(false);
  return true;
#else
  // torch < 2.13: one pair per generator state, shared by its graphs.
  (void)state;
  (void)seed_ptr;
  (void)offset_ptr;
  return false;
#endif
}

void CUDAGraph::capture_begin(MempoolId_t pool, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  auto* gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
  register_generator_state(generator_state_of(gen));
#else
  gen->register_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
#endif

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    mempool_id_ = torch_graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_, mempool_id_, [this](cudaStream_t stream) {
        cudaStreamCaptureStatus status{};
        CaptureId_t stream_capture_id = 0;
        AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
        return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive &&
               stream_capture_id == capture_id_;
      });
  // foundry::resume_allocation_region();
  foundry::start_hook_record();

#if !FOUNDRY_TORCH_PER_CAPTURE_RNG
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->capture_prologue();
  }
#endif

  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  c10::cuda::CUDACachingAllocator::markCaptureBegin(capture_dev_);
#endif

  cudaStreamCaptureStatus status{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  // Per torch 2.13: per-capture RNG state is created once the capture id is
  // known (allocations route to the default pool via internal guards).
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
#if FOUNDRY_TORCH_LAZY_CAPTURE_RNG
    // torch 2.14 creates it on the first RNG op through the at::cuda::CUDAGraph
    // lookup, which cannot find this graph: create it now instead.
    generator_state->get_capture_state(capture_id_, /*create_if_not_found=*/true);
#else
    generator_state->init_capture_state(capture_id_);
#endif
  }
#endif
}

void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_, "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  c10::cuda::CUDACachingAllocator::markCaptureEnd(capture_dev_);
#endif

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  foundry::end_hook_record();
  // foundry::stop_allocation_region();
  allocator_events_ = foundry::save_hook_events_to_json();
  foundry::clear_hook_events();

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
    wholegraph_increments = generator_state->capture_epilogue(capture_id_);
#else
    wholegraph_increments = generator_state->capture_epilogue();
#endif
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
    TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
               "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;

  analyze_captured_graph();

  if (!keep_graph_) {
    instantiate();
    if (!_cuda_graphs_debug) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
    has_graph_ = false;
  }
}

void CUDAGraph::instantiate() {
  TORCH_CHECK(capture_ended_, "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(keep_graph_,
                "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
  }
  graph_exec_ = instantiate_graph_exec(graph_);
  has_graph_exec_ = true;
}

cudaGraphExec_t CUDAGraph::instantiate_graph_exec(cudaGraph_t graph) {
  cudaGraphExec_t exec = nullptr;
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, 0);
    if (inst_err != cudaSuccess) {
      fprintf(stderr, "[foundry INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
              inst_err, cudaGetErrorString(inst_err));
      AT_CUDA_CHECK(inst_err);
    }
#else
  cudaError_t inst_err = cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
  if (inst_err != cudaSuccess) {
    fprintf(stderr, "[foundry INSTANTIATE ERROR] cudaGraphInstantiate FAILED with error %d: %s\n",
            inst_err, cudaGetErrorString(inst_err));
    AT_CUDA_CHECK(inst_err);
  }
#endif
#if !defined(USE_ROCM) || ROCM_VERSION >= 60200
  } else {
    cudaError_t inst_err =
        cudaGraphInstantiateWithFlags(&exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (inst_err != cudaSuccess) {
      fprintf(
          stderr,
          "[foundry INSTANTIATE ERROR] cudaGraphInstantiateWithFlags FAILED with error %d: %s\n",
          inst_err, cudaGetErrorString(inst_err));
      AT_CUDA_CHECK(inst_err);
    }
  }
#endif
  return exec;
}

std::atomic<uint64_t> CUDAGraph::g_member_update_us{0};
std::atomic<uint64_t> CUDAGraph::g_member_inst_us{0};
std::atomic<uint64_t> CUDAGraph::g_member_attr_calls{0};

namespace {

// Kernel-node attributes a rewrite sets, as bits of SharedGraphExec::attrs_maybe_set.
enum : uint8_t {
  kAttrPreferredCluster = 1 << 0,
  kAttrSchedulingPolicy = 1 << 1,
  kAttrCooperative = 1 << 2,
  kAttrPriority = 1 << 3,
  kAttrMemSyncDomain = 1 << 4,
  kAttrMemSyncDomainMap = 1 << 5,
  kAttrCarveout = 1 << 6,
  kAttrAll = 0x7F,
};

uint8_t recorded_attrs(const CUDAGraph::OnDemandNodeUpdate::KernelNodeAttrs& a) {
  return (a.has_preferred_cluster_dim ? kAttrPreferredCluster : 0) |
         (a.clusterSchedulingPolicy >= 0 ? kAttrSchedulingPolicy : 0) |
         (a.cooperative >= 0 ? kAttrCooperative : 0) | (a.priority >= 0 ? kAttrPriority : 0) |
         (a.memSyncDomain >= 0 ? kAttrMemSyncDomain : 0) |
         (a.has_mem_sync_domain_map ? kAttrMemSyncDomainMap : 0) |
         (a.has_shared_mem_carveout ? kAttrCarveout : 0);
}

// SAVE records these attributes only when they are not zero, so one that a graph does not
// record was zero on its node. Before a graph's params go onto the builder node, reset to zero
// the ones the template or an earlier member left set, and a preferred cluster shape that
// differs from this graph's (the new grid is validated against it). `maybe` holds the node's
// attrs_maybe_set bits: only those attributes are read.
void reset_unrecorded_kernel_attrs(CUgraphNode node,
                                   const CUDAGraph::OnDemandNodeUpdate::KernelNodeAttrs& a,
                                   uint8_t& maybe, std::atomic<uint64_t>& calls) {
  const uint8_t recorded = recorded_attrs(a);
  auto reset = [&](uint8_t bit, CUkernelNodeAttrID id, auto keep) {
    if (!(maybe & bit))
      return;
    CUkernelNodeAttrValue cur;
    memset(&cur, 0, sizeof(cur));
    if (cuGraphKernelNodeGetAttribute(node, id, &cur) == CUDA_SUCCESS && !keep(cur)) {
      CUkernelNodeAttrValue zero;
      memset(&zero, 0, sizeof(zero));
      calls++;
      C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(node, id, &zero));
    }
    if (!(recorded & bit))
      maybe &= static_cast<uint8_t>(~bit);
  };
  reset(kAttrPreferredCluster, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION,
        [&](const CUkernelNodeAttrValue& v) {
          const auto& d = v.preferredClusterDim;
          return a.has_preferred_cluster_dim
                     ? d.x == a.preferredClusterDimX && d.y == a.preferredClusterDimY &&
                           d.z == a.preferredClusterDimZ
                     : d.x == 0 && d.y == 0 && d.z == 0;
        });
  if (!(recorded & kAttrSchedulingPolicy))
    reset(kAttrSchedulingPolicy, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
          [](const CUkernelNodeAttrValue& v) { return v.clusterSchedulingPolicyPreference == 0; });
  if (!(recorded & kAttrCooperative))
    reset(kAttrCooperative, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE,
          [](const CUkernelNodeAttrValue& v) { return v.cooperative == 0; });
  if (!(recorded & kAttrPriority))
    reset(kAttrPriority, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY,
          [](const CUkernelNodeAttrValue& v) { return v.priority == 0; });
  if (!(recorded & kAttrMemSyncDomain))
    reset(kAttrMemSyncDomain, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN,
          [](const CUkernelNodeAttrValue& v) { return v.memSyncDomain == 0; });
  if (!(recorded & kAttrMemSyncDomainMap))
    reset(kAttrMemSyncDomainMap, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP,
          [](const CUkernelNodeAttrValue& v) {
            return v.memSyncDomainMap.default_ == 0 && v.memSyncDomainMap.remote == 0;
          });
  if (!(recorded & kAttrCarveout))
    reset(kAttrCarveout, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
          [](const CUkernelNodeAttrValue& v) { return v.sharedMemCarveout == 0; });
}

}  // namespace

void CUDAGraph::apply_on_demand_updates() {
  auto& shared = on_demand_data_->shared_exec;
  if (shared->attrs_maybe_set.size() != shared->ordered_nodes.size())
    shared->attrs_maybe_set.assign(shared->ordered_nodes.size(), kAttrAll);
  for (size_t i = 0; i < on_demand_data_->updates.size(); ++i) {
    auto& u = on_demand_data_->updates[i];
    CUgraphNode node = shared->ordered_nodes[i];
    switch (u.type) {
      case OnDemandNodeUpdate::Kernel: {
        reset_unrecorded_kernel_attrs(node, u.kernel_attrs, shared->attrs_maybe_set[i],
                                      g_member_attr_calls);
        // Update kernel params on the graph node (not exec)
        // A member may launch a different cluster shape than its template (deep_gemm picks the
        // cluster size by M; FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0 puts such graphs in one group).
        // cuGraphKernelNodeSetParams validates the new grid against the node's *current* cluster
        // dims, so neutralise them to 1x1x1 first when they differ; the member's own dims are
        // applied right after the params update below.
        if (u.kernel_attrs.has_cluster_dim) {
          CUkernelNodeAttrValue cur;
          memset(&cur, 0, sizeof(cur));
          const unsigned wx = u.kernel_attrs.clusterDimX > 0 ? u.kernel_attrs.clusterDimX : 1;
          const unsigned wy = u.kernel_attrs.clusterDimY > 0 ? u.kernel_attrs.clusterDimY : 1;
          const unsigned wz = u.kernel_attrs.clusterDimZ > 0 ? u.kernel_attrs.clusterDimZ : 1;
          if (cuGraphKernelNodeGetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION,
                                            &cur) == CUDA_SUCCESS &&
              (cur.clusterDim.x != wx || cur.clusterDim.y != wy || cur.clusterDim.z != wz)) {
            CUkernelNodeAttrValue one;
            memset(&one, 0, sizeof(one));
            one.clusterDim.x = one.clusterDim.y = one.clusterDim.z = 1;
            g_member_attr_calls++;
            cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &one);
          }
        }
        ensure_dynamic_smem_optin(u.kernel_params, capture_dev_, "ON-DEMAND");
        if (u.kernel_attrs.has_cluster_dim)
          ensure_cluster_size_optin(u.kernel_params, capture_dev_, u.kernel_attrs.clusterDimX,
                                    u.kernel_attrs.clusterDimY, u.kernel_attrs.clusterDimZ,
                                    "ON-DEMAND");
        CUresult sp = cuGraphKernelNodeSetParams(node, &u.kernel_params);
        if (sp != CUDA_SUCCESS && u.kernel_params.kern && !u.kernel_params.func) {
          // Re-targeting a node to another CUkernel can be rejected; retry
          // with the kernel's per-context CUfunction instead.
          CUDA_KERNEL_NODE_PARAMS alt = u.kernel_params;
          alt.kern = nullptr;
          if (cuKernelGetFunction(&alt.func, u.kernel_params.kern) == CUDA_SUCCESS && alt.func) {
            ensure_dynamic_smem_optin(alt, capture_dev_, "ON-DEMAND");
            if (u.kernel_attrs.has_cluster_dim)
              ensure_cluster_size_optin(alt, capture_dev_, u.kernel_attrs.clusterDimX,
                                        u.kernel_attrs.clusterDimY, u.kernel_attrs.clusterDimZ,
                                        "ON-DEMAND");
            CUresult sp2 = cuGraphKernelNodeSetParams(node, &alt);
            fprintf(stderr,
                    "[foundry ON-DEMAND] graph %d node %zu: kern-based SetParams failed (%d), "
                    "func-based retry -> %d\n",
                    on_demand_data_->graph_id, i, (int)sp, (int)sp2);
            if (sp2 == CUDA_SUCCESS) {
              u.kernel_params = alt;
              sp = sp2;
            }
          }
        }
        if (sp != CUDA_SUCCESS) {
          const char* err = nullptr;
          cuGetErrorString(sp, &err);
          CUDA_KERNEL_NODE_PARAMS cur{};
          cuGraphKernelNodeGetParams(node, &cur);
          fprintf(stderr,
                  "[foundry ON-DEMAND ERROR] graph %d (%s) node %zu: cuGraphKernelNodeSetParams "
                  "failed: %d (%s). new: func=%p kern=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u "
                  "| template node: func=%p kern=%p grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\n",
                  on_demand_data_->graph_id, on_demand_data_->graph_name.c_str(), i, (int)sp,
                  err ? err : "?", (void*)u.kernel_params.func, (void*)u.kernel_params.kern,
                  u.kernel_params.gridDimX, u.kernel_params.gridDimY, u.kernel_params.gridDimZ,
                  u.kernel_params.blockDimX, u.kernel_params.blockDimY, u.kernel_params.blockDimZ,
                  u.kernel_params.sharedMemBytes, (void*)cur.func, (void*)cur.kern, cur.gridDimX,
                  cur.gridDimY, cur.gridDimZ, cur.blockDimX, cur.blockDimY, cur.blockDimZ,
                  cur.sharedMemBytes);
        }
        C10_CUDA_DRIVER_CHECK(sp);

        // Update kernel node attributes that may differ across batch sizes
        auto& a = u.kernel_attrs;
        if (a.has_cluster_dim) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.clusterDim.x = a.clusterDimX > 0 ? a.clusterDimX : 1;
          attr.clusterDim.y = a.clusterDimY > 0 ? a.clusterDimY : 1;
          attr.clusterDim.z = a.clusterDimZ > 0 ? a.clusterDimZ : 1;
          g_member_attr_calls++;
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &attr));
        }
        if (a.has_preferred_cluster_dim) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.preferredClusterDim.x = a.preferredClusterDimX;
          attr.preferredClusterDim.y = a.preferredClusterDimY;
          attr.preferredClusterDim.z = a.preferredClusterDimZ;
          g_member_attr_calls++;
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr));
        }
        if (a.clusterSchedulingPolicy >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.clusterSchedulingPolicyPreference =
              static_cast<CUclusterSchedulingPolicy>(a.clusterSchedulingPolicy);
          g_member_attr_calls++;
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr));
        }
        if (a.cooperative >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.cooperative = a.cooperative;
          C10_CUDA_DRIVER_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE, &attr));
        }
        if (a.priority >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.priority = a.priority;
          C10_CUDA_DRIVER_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr));
        }
        if (a.memSyncDomain >= 0) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.memSyncDomain = static_cast<CUlaunchMemSyncDomain>(a.memSyncDomain);
          C10_CUDA_DRIVER_CHECK(
              cuGraphKernelNodeSetAttribute(node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN, &attr));
        }
        if (a.has_mem_sync_domain_map) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.memSyncDomainMap.default_ = a.memSyncDomainMapDefault;
          attr.memSyncDomainMap.remote = a.memSyncDomainMapRemote;
          g_member_attr_calls++;
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr));
        }
        if (a.has_shared_mem_carveout) {
          CUkernelNodeAttrValue attr;
          memset(&attr, 0, sizeof(attr));
          attr.sharedMemCarveout = a.sharedMemCarveout;
          g_member_attr_calls++;
          C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetAttribute(
              node, CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr));
        }
        shared->attrs_maybe_set[i] |= recorded_attrs(a);
        break;
      }
      case OnDemandNodeUpdate::Memset:
        C10_CUDA_DRIVER_CHECK(cuGraphMemsetNodeSetParams(node, &u.memset_params));
        break;
      case OnDemandNodeUpdate::Memcpy:
        C10_CUDA_DRIVER_CHECK(cuGraphMemcpyNodeSetParams(node, &u.memcpy_params));
        break;
      case OnDemandNodeUpdate::EventRecord:
        C10_CUDA_DRIVER_CHECK(cuGraphEventRecordNodeSetEvent(node, u.event));
        break;
      case OnDemandNodeUpdate::EventWait:
        C10_CUDA_DRIVER_CHECK(cuGraphEventWaitNodeSetEvent(node, u.event));
        break;
      case OnDemandNodeUpdate::Empty:
        break;
    }
  }
}

CUDAGraph::OnDemandData::~OnDemandData() {
  if (owns_exec && own_exec) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(own_exec)));
  }
}

void CUDAGraph::bind_shared_exec() {
  TORCH_CHECK(on_demand_data_ && on_demand_data_->shared_exec,
              "bind_shared_exec: graph is not linked to a shared exec");
  auto& shared = on_demand_data_->shared_exec;
  // Template (the shared exec was instantiated with these params), or share
  // mode: replay switches the shared exec to this graph before launch. An
  // eager member owns the exec Phase 2 instantiated for it instead.
  TORCH_CHECK(shared->share || shared->current_params_id == on_demand_data_->graph_id,
              "bind_shared_exec: eager member ", on_demand_data_->graph_name,
              " cannot launch the template's exec");
  on_demand_data_->own_exec = shared->exec;
  on_demand_data_->owns_exec = false;
}

void CUDAGraph::rewrite_shared_graph_for_member() {
  auto t_upd = std::chrono::steady_clock::now();
  apply_on_demand_updates();
  on_demand_data_->shared_exec->current_params_id = on_demand_data_->graph_id;
  g_member_update_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - t_upd)
                            .count();
}

void CUDAGraph::instantiate_member_exec() {
  auto& shared = on_demand_data_->shared_exec;
  auto t_inst = std::chrono::steady_clock::now();
  cudaGraphExec_t exec = nullptr;
  cudaError_t inst_err =
      cudaGraphInstantiate(&exec, reinterpret_cast<cudaGraph_t>(shared->graph), 0);
  auto t_end = std::chrono::steady_clock::now();
  g_member_inst_us += std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_inst).count();
  if (inst_err != cudaSuccess) {
    fprintf(stderr,
            "[foundry ON-DEMAND ERROR] cudaGraphInstantiate FAILED for graph %d (%s): %d (%s)\n",
            on_demand_data_->graph_id, on_demand_data_->graph_name.c_str(), (int)inst_err,
            cudaGetErrorString(inst_err));
    AT_CUDA_CHECK(inst_err);
  }
  on_demand_data_->own_exec = reinterpret_cast<CUgraphExec>(exec);
  on_demand_data_->owns_exec = true;
}

void CUDAGraph::replay() {
  // Eager: every graph launches its own exec (the template's is the shared one).
  // Share (FOUNDRY_ENABLE_EXEC_SHARE=1): a replay of another member than the
  // last one rewrites the builder graph to this member, updates the shared exec
  // from it (cuGraphExecUpdate) and uploads it; a repeat of the same member
  // launches directly. No stream or device synchronization.
  if (on_demand_data_) {
    c10::OptionalDeviceGuard device_guard{c10::Device(c10::kCUDA, capture_dev_)};
    auto& shared = *on_demand_data_->shared_exec;
    std::unique_lock<std::mutex> replay_lock(shared.replay_mutex, std::defer_lock);
    if (shared.share) {
      // Held through the launch submission: one member's params on the exec at a time.
      replay_lock.lock();
      TORCH_CHECK(shared.valid, "exec share: an earlier switch of this template's exec failed (",
                  on_demand_data_->graph_name, ")");
      if (shared.current_params_id != on_demand_data_->graph_id) {
        using clk = std::chrono::steady_clock;
        auto ns_since = [](clk::time_point t) {
          return static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count());
        };
        shared.valid = false;
        auto t = clk::now();
        apply_on_demand_updates();
        shared.rewrite_ns += ns_since(t);
        t = clk::now();
        CUgraphExecUpdateResultInfo info{};
        if (cgp_exec_update) {
          // cudagraph-patch: prepare + cuGraphExecUpdate_v2 + finish on a registered exec.
          // Until the patch passes unregistered / inert cases through to the real update,
          // any nonzero return means the exec was not (correctly) updated.
          const int rc = cgp_exec_update(shared.exec, shared.graph, &info);
          shared.update_ns += ns_since(t);
          TORCH_CHECK(rc == 0 && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
                      "exec share: cgp_exec_update failed for ", on_demand_data_->graph_name,
                      " (rc ", rc, ", update result ", static_cast<int>(info.result), ")");
        } else {
          CUresult rc = cuGraphExecUpdate_v2(shared.exec, shared.graph, &info);
          shared.update_ns += ns_since(t);
          TORCH_CHECK(rc == CUDA_SUCCESS && info.result == CU_GRAPH_EXEC_UPDATE_SUCCESS,
                      "exec share: cuGraphExecUpdate failed for ", on_demand_data_->graph_name,
                      " (CUresult ", static_cast<int>(rc), ", update result ",
                      static_cast<int>(info.result), ")");
        }
        t = clk::now();
        C10_CUDA_DRIVER_CHECK(cuGraphUpload(shared.exec, at::cuda::getCurrentCUDAStream()));
        shared.upload_ns += ns_since(t);
        shared.current_params_id = on_demand_data_->graph_id;
        shared.updates++;
        shared.valid = true;
      }
    }
    TORCH_CHECK(on_demand_data_->own_exec, "replay: ", on_demand_data_->graph_name,
                " has no exec (LOAD Phase 2 did not finish)");
    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
      generator_state->replay_prologue(capture_id_, wholegraph_increments);
#else
      generator_state->replay_prologue(wholegraph_increments);
#endif
    }
    AT_CUDA_CHECK(cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(on_demand_data_->own_exec),
                                  at::cuda::getCurrentCUDAStream()));
    return;
  }

#ifdef FOUNDRY_DEBUG
  fprintf(stderr, "[foundry DEBUG] replay: NO on_demand_data_, using direct exec path\n");
#endif

  TORCH_CHECK(capture_ended_ || has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture or load.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
    generator_state->replay_prologue(capture_id_, wholegraph_increments);
#else
    generator_state->replay_prologue(wholegraph_increments);
#endif
  }
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if defined(CUDA_VERSION) || defined(USE_ROCM)
  if (_cuda_graphs_debug || keep_graph_) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(
          cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose));
      if (!keep_graph_) {
        AT_CUDA_CHECK(cudaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
#else
  TORCH_CHECK(false,
              "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

cudaGraph_t CUDAGraph::raw_cuda_graph() {
  TORCH_CHECK(keep_graph_,
              "You cannot access the raw cudaGraph_t instance unless CUDAGraph was initialized "
              "with keep_graph=true");
  TORCH_CHECK(has_graph_,
              "You cannot access the raw cudaGraph_t instance until capture_end() has been called");
  return graph_;
}

cudaGraphExec_t CUDAGraph::raw_cuda_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw cudaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
}

CUDAGraph::SharedGraphExec::~SharedGraphExec() {
  if (exec) {
    if (share && cgp_exec_forget)
      cgp_exec_forget(exec);  // an exec it never registered returns CGP_E_UNKNOWN_EXEC
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(reinterpret_cast<cudaGraphExec_t>(exec)));
  }
  if (graph) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph)));
  }
}

void CUDAGraph::instantiate_shared_template(SharedGraphExec& shared) {
  if (!cgp_exec_register_begin || !cgp_exec_register_end) {
    shared.exec = reinterpret_cast<CUgraphExec>(
        instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(shared.graph)));
  } else {
    // cudagraph-patch records the exec's descriptors during this instantiate (same thread).
    // Census of the public graph for cgp_exec_register_expect: CUfunction per kernel
    // node, memcpy nodes, programmatic (PDL) edges.
    std::vector<void*> functions;
    unsigned memcpy_nodes = 0;
    for (CUgraphNode node : shared.ordered_nodes) {
      CUgraphNodeType type;
      C10_CUDA_DRIVER_CHECK(cuGraphNodeGetType(node, &type));
      if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
        CUDA_KERNEL_NODE_PARAMS params{};
        C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(node, &params));
        if (!params.func && params.kern)
          C10_CUDA_DRIVER_CHECK(cuKernelGetFunction(&params.func, params.kern));
        TORCH_CHECK(params.func, "exec share: cannot resolve a kernel node's CUfunction");
        functions.push_back(reinterpret_cast<void*>(params.func));
      } else {
        memcpy_nodes += type == CU_GRAPH_NODE_TYPE_MEMCPY;
      }
    }
    size_t count = 0;
    C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(shared.graph, nullptr, nullptr, nullptr, &count));
    std::vector<CUgraphNode> from(count), to(count);
    std::vector<CUgraphEdgeData> edges(count);
    C10_CUDA_DRIVER_CHECK(
        cuGraphGetEdges_v2(shared.graph, from.data(), to.data(), edges.data(), &count));
    unsigned pdl_edges = 0;
    for (const auto& edge : edges)
      pdl_edges += edge.type == CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC;
    const unsigned kernels = static_cast<unsigned>(functions.size());
    int rc = cgp_exec_register_begin();
    if (rc)
      fprintf(stderr, "[foundry] exec share: cgp_exec_register_begin failed (%d)\n", rc);
    try {
      if (cgp_exec_register_expect) {
        cgp_census_t census{};
        census.struct_size = sizeof(census);
        census.functions = functions.data();
        census.function_count = kernels;
        census.pdl_edges = pdl_edges;
        census.memcpy_nodes = memcpy_nodes;
        rc = cgp_exec_register_expect(&census);
        if (rc)
          fprintf(stderr, "[foundry] exec share: cgp_exec_register_expect failed (%d)\n", rc);
      }
      shared.exec = reinterpret_cast<CUgraphExec>(
          instantiate_graph_exec(reinterpret_cast<cudaGraph_t>(shared.graph)));
    } catch (...) {
      cgp_exec_register_end(nullptr, 0);  // cancels the scope
      throw;
    }
    rc = cgp_exec_register_end(shared.exec, kernels);
    if (rc)  // the first switch of this exec then fails at cgp_exec_update
      fprintf(stderr, "[foundry] exec share: cgp_exec_register_end failed (%d)\n", rc);
  }
  // Upload before the first update: an update ahead of an exec's first upload
  // demotes every changed node (docs/exec-update-penalty.md).
  C10_CUDA_DRIVER_CHECK(cuGraphUpload(shared.exec, nullptr));
}

std::string CUDAGraph::exec_share_stats() const {
  boost::json::object out;
  if (on_demand_data_ && on_demand_data_->shared_exec) {
    auto& shared = *on_demand_data_->shared_exec;
    std::lock_guard<std::mutex> lock(shared.replay_mutex);
    out["enabled"] = shared.share;
    out["route"] = exec_share::patched_route() ? "cgp" : "cuGraphExecUpdate";
    out["valid"] = shared.valid;
    out["exec"] = reinterpret_cast<uintptr_t>(shared.exec);
    out["graph_id"] = on_demand_data_->graph_id;
    out["current_graph_id"] = shared.current_params_id;
    out["updates"] = shared.updates;
    out["rewrite_ns"] = shared.rewrite_ns;
    out["update_ns"] = shared.update_ns;
    out["upload_ns"] = shared.upload_ns;
  } else {
    out["enabled"] = false;
  }
  return boost::json::serialize(out);
}

void CUDAGraph::transfer_to_shared_exec(std::shared_ptr<SharedGraphExec> shared,
                                        GraphTemplate&& tmpl) {
  shared->exec = reinterpret_cast<CUgraphExec>(graph_exec_);
  shared->graph = reinterpret_cast<CUgraph>(graph_);
  shared->ordered_nodes = std::move(tmpl.ordered_nodes);
  // This graph no longer owns the CUgraph/CUgraphExec
  graph_exec_ = nullptr;
  has_graph_exec_ = false;
  graph_ = nullptr;
  has_graph_ = false;
}

void CUDAGraph::OnDemandNodeUpdate::fixup_pointers() {
  if (type != Kernel)
    return;
  // Fix kernel param pointers
  if (!kernel_param_data.empty()) {
    kernel_param_ptrs.resize(kernel_param_data.size());
    for (size_t j = 0; j < kernel_param_data.size(); ++j)
      kernel_param_ptrs[j] = kernel_param_data[j].data();
    kernel_params.kernelParams = kernel_param_ptrs.data();
  }
  // Fix extra config — rebuild with stable pointers
  if (!arg_buffer.empty()) {
    extra_config.clear();
    extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
    extra_config.push_back(arg_buffer.data());
    extra_config.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
    extra_config.push_back(&arg_buffer_size);
    extra_config.push_back(CU_LAUNCH_PARAM_END);
    kernel_params.extra = extra_config.data();
  }
}

void CUDAGraph::reset() {
  if (capture_ended_) {
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

MempoolId_t CUDAGraph::pool() {
  TORCH_CHECK(capture_ended_, "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
#if FOUNDRY_TORCH_PER_CAPTURE_RNG
  if (capture_id_ != static_cast<CaptureId_t>(-1)) {
    for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
      generator_state->remove_capture_state(capture_id_);
    }
  }
#else
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->unregister_graph(reinterpret_cast<at::cuda::CUDAGraph*>(this));
  }
#endif
  reset();

#if (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  if (capture_dev_ != UNDEFINED_DEVICE) {
    AT_CUDA_CHECK(cudaSetDevice(capture_dev_));
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}

void CUDAGraph::analyze_captured_graph() {
  TORCH_CHECK(has_graph_, "analyze_captured_graph() called before the graph is captured");

  graph_nodes.clear();
  graph_dependencies.clear();

  CUgraph cuGraph = reinterpret_cast<CUgraph>(graph_);

  size_t numNodes = 0;
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nullptr, &numNodes));

  TORCH_CHECK(numNodes > 0, "Graph contains no nodes");

  std::vector<CUgraphNode> nodes(numNodes);
  C10_CUDA_DRIVER_CHECK(cuGraphGetNodes(cuGraph, nodes.data(), &numNodes));

  for (size_t i = 0; i < numNodes; ++i) {
    CUgraphNodeType nodeType;
    C10_CUDA_DRIVER_CHECK(cuGraphNodeGetType(nodes[i], &nodeType));

    GraphNode graphNode;
    graphNode.index = i;
    graphNode.node = nodes[i];

    if (nodeType == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(nodes[i], &params));

      KernelNodeMetadata metadata;
      metadata.blockDimX = params.blockDimX;
      metadata.blockDimY = params.blockDimY;
      metadata.blockDimZ = params.blockDimZ;
      metadata.gridDimX = params.gridDimX;
      metadata.gridDimY = params.gridDimY;
      metadata.gridDimZ = params.gridDimZ;
      metadata.func = params.func;
      metadata.kern = params.kern;
      metadata.ctx = params.ctx;
      metadata.sharedMemBytes = params.sharedMemBytes;

      {
        CUkernelNodeAttrValue attr_val;
        CUresult res = cuGraphKernelNodeGetAttribute(
            nodes[i], CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION, &attr_val);
        if (res == CUDA_SUCCESS) {
          metadata.node_attrs.attr_query_available = true;
          metadata.node_attrs.has_cluster_dim = true;
          metadata.node_attrs.clusterDimX = attr_val.clusterDim.x;
          metadata.node_attrs.clusterDimY = attr_val.clusterDim.y;
          metadata.node_attrs.clusterDimZ = attr_val.clusterDim.z;
        } else {
          metadata.node_attrs.attr_query_available = false;
        }

        if (metadata.node_attrs.attr_query_available) {
          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_cluster_dim = true;
            metadata.node_attrs.preferredClusterDimX = attr_val.preferredClusterDim.x;
            metadata.node_attrs.preferredClusterDimY = attr_val.preferredClusterDim.y;
            metadata.node_attrs.preferredClusterDimZ = attr_val.preferredClusterDim.z;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cluster_scheduling_policy = true;
            metadata.node_attrs.clusterSchedulingPolicyPreference =
                static_cast<int>(attr_val.clusterSchedulingPolicyPreference);
          }

          res = cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE,
                                              &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_cooperative = true;
            metadata.node_attrs.cooperative = attr_val.cooperative;
          }

          res =
              cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PRIORITY, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_priority = true;
            metadata.node_attrs.priority = attr_val.priority;
          }

          res = cuGraphKernelNodeGetAttribute(nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN,
                                              &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain = true;
            metadata.node_attrs.memSyncDomain = attr_val.memSyncDomain;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_mem_sync_domain_map = true;
            metadata.node_attrs.memSyncDomainMapDefault = attr_val.memSyncDomainMap.default_;
            metadata.node_attrs.memSyncDomainMapRemote = attr_val.memSyncDomainMap.remote;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_preferred_shared_mem_carveout = true;
            metadata.node_attrs.preferredSharedMemCarveout = attr_val.sharedMemCarveout;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_access_policy_window = true;
            metadata.node_attrs.accessPolicyWindowBasePtr = attr_val.accessPolicyWindow.base_ptr;
            metadata.node_attrs.accessPolicyWindowNumBytes = attr_val.accessPolicyWindow.num_bytes;
            metadata.node_attrs.accessPolicyWindowHitRatio = attr_val.accessPolicyWindow.hitRatio;
            metadata.node_attrs.accessPolicyWindowHitProp = attr_val.accessPolicyWindow.hitProp;
            metadata.node_attrs.accessPolicyWindowMissProp = attr_val.accessPolicyWindow.missProp;
          }

          res = cuGraphKernelNodeGetAttribute(
              nodes[i], CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_device_updatable = true;
            metadata.node_attrs.deviceUpdatable =
                attr_val.deviceUpdatableKernelNode.deviceUpdatable;
            metadata.node_attrs.deviceUpdatableNode = attr_val.deviceUpdatableKernelNode.devNode;
          }

          // Programmatic dependent launch: kernels captured from PDL launches
          // (cudaLaunchKernelExC with PROGRAMMATIC_STREAM_SERIALIZATION, e.g.
          // FA3 fwd/combine on sm90) carry this node attribute; a rebuilt
          // node without it loses the launch overlap the attribute permits.
          res = cuGraphKernelNodeGetAttribute(
              nodes[i],
              static_cast<CUkernelNodeAttrID>(
                  CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION),
              &attr_val);
          if (res == CUDA_SUCCESS) {
            metadata.node_attrs.has_programmatic_stream_serialization = true;
            metadata.node_attrs.programmaticStreamSerializationAllowed =
                attr_val.programmaticStreamSerializationAllowed;
          }
        }
      }

      metadata.num_params = 0;
      size_t offset, size;
      while (cuFuncGetParamInfo(params.func, metadata.num_params, &offset, &size) == CUDA_SUCCESS) {
        metadata.offset_and_sizes.emplace_back(offset, size);
        metadata.num_params++;
      }

      if (params.kernelParams) {
        size_t totalSize = 0;
        for (const auto& [offset, size] : metadata.offset_and_sizes) {
          totalSize += size;
        }

        if (totalSize > 0) {
          metadata.kernelParams = static_cast<void**>(malloc(metadata.num_params * sizeof(void*)));
          for (int j = 0; j < metadata.num_params; ++j) {
            metadata.kernelParams[j] = malloc(std::get<1>(metadata.offset_and_sizes[j]));
            memcpy(metadata.kernelParams[j], params.kernelParams[j],
                   std::get<1>(metadata.offset_and_sizes[j]));
          }
        }
      }

      if (params.extra) {
        void** config = params.extra;
        size_t argBufferSize = 0;
        void* argBufferPtr = nullptr;

        int idx;
        for (idx = 0; config[idx] != CU_LAUNCH_PARAM_END; idx++) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            argBufferPtr = config[idx + 1];
            idx++;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            idx++;
          }
        }
        int configItems = idx + 1;

        metadata.extraSize = configItems * sizeof(void*);
        metadata.extra = malloc(metadata.extraSize);
        void** newConfig = static_cast<void**>(metadata.extra);

        if (argBufferSize > 0) {
          metadata.argBufferSize = static_cast<size_t*>(malloc(sizeof(size_t)));
          *metadata.argBufferSize = argBufferSize;
          metadata.argBuffer = malloc(argBufferSize);
          TORCH_INTERNAL_ASSERT(argBufferPtr != nullptr);
          memcpy(metadata.argBuffer, argBufferPtr, argBufferSize);
        }

        for (int idx = 0; idx < configItems; idx++) {
          if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBuffer;
            idx++;
          } else if (idx < configItems - 1 && config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            newConfig[idx] = config[idx];
            newConfig[idx + 1] = metadata.argBufferSize;
            idx++;
          } else if (idx == configItems - 1) {
            newConfig[idx] = CU_LAUNCH_PARAM_END;
          } else {
            newConfig[idx] = config[idx];
          }
        }
      }

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphMemsetNodeGetParams(nodes[i], &params));

      MemsetNodeMetadata metadata;
      metadata.dst = params.dst;
      metadata.elementSize = params.elementSize;
      metadata.height = params.height;
      metadata.pitch = params.pitch;
      metadata.value = params.value;
      metadata.width = params.width;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D params;
      memset(&params, 0, sizeof(params));
      C10_CUDA_DRIVER_CHECK(cuGraphMemcpyNodeGetParams(nodes[i], &params));

      MemcpyNodeMetadata metadata;
      metadata.Depth = params.Depth;
      metadata.Height = params.Height;
      metadata.WidthInBytes = params.WidthInBytes;
      metadata.dstArray = params.dstArray;
      metadata.dstDevice = params.dstDevice;
      metadata.dstHeight = params.dstHeight;
      metadata.dstHost = params.dstHost;
      metadata.dstLOD = params.dstLOD;
      metadata.dstMemoryType = params.dstMemoryType;
      metadata.dstPitch = params.dstPitch;
      metadata.dstXInBytes = params.dstXInBytes;
      metadata.dstY = params.dstY;
      metadata.dstZ = params.dstZ;
      metadata.reserved0 = params.reserved0;
      metadata.reserved1 = params.reserved1;
      metadata.srcArray = params.srcArray;
      metadata.srcDevice = params.srcDevice;
      metadata.srcHeight = params.srcHeight;
      metadata.srcHost = params.srcHost;
      metadata.srcLOD = params.srcLOD;
      metadata.srcMemoryType = params.srcMemoryType;
      metadata.srcPitch = params.srcPitch;
      metadata.srcXInBytes = params.srcXInBytes;
      metadata.srcY = params.srcY;
      metadata.srcZ = params.srcZ;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
      CUevent event;
      C10_CUDA_DRIVER_CHECK(cuGraphEventRecordNodeGetEvent(nodes[i], &event));

      EventRecordNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      CUevent event;
      C10_CUDA_DRIVER_CHECK(cuGraphEventWaitNodeGetEvent(nodes[i], &event));

      EventWaitNodeMetadata metadata;
      metadata.event = event;

      graphNode.metadata = std::move(metadata);
    } else if (nodeType == CU_GRAPH_NODE_TYPE_EMPTY) {
      EmptyNodeMetadata metadata;

      graphNode.metadata = std::move(metadata);
    } else {
      TORCH_CHECK(false, "Graph contains unsupported node type!");
    }

    graph_nodes.push_back(std::move(graphNode));
  }

  size_t numEdges = 0;
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#else
  C10_CUDA_DRIVER_CHECK(cuGraphGetEdges_v2(cuGraph, nullptr, nullptr, nullptr, &numEdges));
#endif

  if (numEdges > 0) {
    std::vector<CUgraphNode> from_nodes(numEdges);
    std::vector<CUgraphNode> to_nodes(numEdges);
    std::vector<CUgraphEdgeData> edges(numEdges);
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 13000)
    C10_CUDA_DRIVER_CHECK(
        cuGraphGetEdges(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
#else
    C10_CUDA_DRIVER_CHECK(
        cuGraphGetEdges_v2(cuGraph, from_nodes.data(), to_nodes.data(), edges.data(), &numEdges));
#endif
    std::unordered_map<CUgraphNode, int> node_to_index;
    for (size_t i = 0; i < nodes.size(); i++) {
      node_to_index[nodes[i]] = i;
    }

    for (size_t i = 0; i < numEdges; i++) {
      auto from_it = node_to_index.find(from_nodes[i]);
      auto to_it = node_to_index.find(to_nodes[i]);

      if (from_it != node_to_index.end() && to_it != node_to_index.end()) {
        GraphDependency dep;
        dep.from_index = from_it->second;
        dep.to_index = to_it->second;
        dep.from_port = edges[i].from_port;
        dep.to_port = edges[i].to_port;
        dep.edge_type = edges[i].type;
        graph_dependencies.push_back(dep);
      }
    }
  }
}

static boost::json::object serialize_tensor_metadata(const at::Tensor& tensor) {
  namespace json = boost::json;
  json::object obj;

  obj["data_ptr"] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()));

  json::array sizes_arr;
  for (auto s : tensor.sizes()) {
    sizes_arr.push_back(static_cast<int64_t>(s));
  }
  obj["sizes"] = sizes_arr;

  json::array strides_arr;
  for (auto s : tensor.strides()) {
    strides_arr.push_back(static_cast<int64_t>(s));
  }
  obj["strides"] = strides_arr;

  obj["dtype"] = static_cast<int>(tensor.scalar_type());
  obj["device_type"] = static_cast<int>(tensor.device().type());
  obj["device_index"] = tensor.device().index();
  obj["requires_grad"] = tensor.requires_grad();
  obj["numel"] = static_cast<int64_t>(tensor.numel());
  obj["element_size"] = static_cast<int64_t>(tensor.element_size());

  return obj;
}

static at::Tensor reconstruct_tensor_from_metadata(const boost::json::object& obj) {
  uintptr_t data_ptr = static_cast<uintptr_t>(obj.at("data_ptr").to_number<uint64_t>());

  const auto& sizes_arr = obj.at("sizes").as_array();
  std::vector<int64_t> sizes;
  for (const auto& s : sizes_arr) {
    sizes.push_back(s.to_number<int64_t>());
  }

  const auto& strides_arr = obj.at("strides").as_array();
  std::vector<int64_t> strides;
  for (const auto& s : strides_arr) {
    strides.push_back(s.to_number<int64_t>());
  }

  auto dtype = static_cast<c10::ScalarType>(obj.at("dtype").to_number<int>());
  auto device_type = static_cast<c10::DeviceType>(obj.at("device_type").to_number<int>());
  auto device_index = static_cast<c10::DeviceIndex>(obj.at("device_index").to_number<int>());
  bool requires_grad = obj.at("requires_grad").as_bool();

  auto options = at::TensorOptions()
                     .dtype(dtype)
                     .device(c10::Device(device_type, device_index))
                     .requires_grad(requires_grad);

  auto tensor = at::from_blob(reinterpret_cast<void*>(data_ptr), sizes, strides, options);

  return tensor;
}

void CUDAGraph::save(const std::string& json_path, const OutputTensors& output_tensors,
                     OutputTensorType output_type) {
  TORCH_CHECK(!graph_nodes.empty(), "Graph hasn't been captured yet or has no nodes");

  set_pack_fatbins_on_exit(true);

  namespace json = boost::json;
  json::object root;
  json::array nodes_array;

  std::unordered_map<CUevent, int> event_to_id;
  int next_event_id = 0;

  for (const auto& graphNode : graph_nodes) {
    json::object node_obj;
    node_obj["id"] = graphNode.index;

    if (std::holds_alternative<KernelNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<KernelNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "KernelNode";

      json::object params;
      params["blockDimX"] = metadata.blockDimX;
      params["blockDimY"] = metadata.blockDimY;
      params["blockDimZ"] = metadata.blockDimZ;
      params["gridDimX"] = metadata.gridDimX;
      params["gridDimY"] = metadata.gridDimY;
      params["gridDimZ"] = metadata.gridDimZ;
      params["sharedMemBytes"] = metadata.sharedMemBytes;

      // Only save non-default kernel node attributes to avoid unnecessary
      // cuGraphKernelNodeSetAttribute calls on load (each call acquires
      // the driver context lock).
      {
        json::object kernel_node_attrs;
        kernel_node_attrs["attrQueryAvailable"] = metadata.node_attrs.attr_query_available;
        // cluster_dim: default is 1x1x1, only save if any dimension > 1
        if (metadata.node_attrs.has_cluster_dim &&
            (metadata.node_attrs.clusterDimX > 1 || metadata.node_attrs.clusterDimY > 1 ||
             metadata.node_attrs.clusterDimZ > 1)) {
          kernel_node_attrs["clusterDimX"] = metadata.node_attrs.clusterDimX;
          kernel_node_attrs["clusterDimY"] = metadata.node_attrs.clusterDimY;
          kernel_node_attrs["clusterDimZ"] = metadata.node_attrs.clusterDimZ;
        }
        if (metadata.node_attrs.has_preferred_cluster_dim &&
            (metadata.node_attrs.preferredClusterDimX > 0 ||
             metadata.node_attrs.preferredClusterDimY > 0 ||
             metadata.node_attrs.preferredClusterDimZ > 0)) {
          kernel_node_attrs["preferredClusterDimX"] = metadata.node_attrs.preferredClusterDimX;
          kernel_node_attrs["preferredClusterDimY"] = metadata.node_attrs.preferredClusterDimY;
          kernel_node_attrs["preferredClusterDimZ"] = metadata.node_attrs.preferredClusterDimZ;
        }
        if (metadata.node_attrs.has_cluster_scheduling_policy &&
            metadata.node_attrs.clusterSchedulingPolicyPreference != 0) {
          kernel_node_attrs["clusterSchedulingPolicyPreference"] =
              metadata.node_attrs.clusterSchedulingPolicyPreference;
        }
        if (metadata.node_attrs.has_cooperative && metadata.node_attrs.cooperative != 0) {
          kernel_node_attrs["cooperative"] = metadata.node_attrs.cooperative;
        }
        if (metadata.node_attrs.has_priority && metadata.node_attrs.priority != 0) {
          kernel_node_attrs["priority"] = metadata.node_attrs.priority;
        }
        if (metadata.node_attrs.has_mem_sync_domain && metadata.node_attrs.memSyncDomain != 0) {
          kernel_node_attrs["memSyncDomain"] = metadata.node_attrs.memSyncDomain;
        }
        if (metadata.node_attrs.has_mem_sync_domain_map &&
            (metadata.node_attrs.memSyncDomainMapDefault != 0 ||
             metadata.node_attrs.memSyncDomainMapRemote != 0)) {
          kernel_node_attrs["memSyncDomainMapDefault"] =
              metadata.node_attrs.memSyncDomainMapDefault;
          kernel_node_attrs["memSyncDomainMapRemote"] = metadata.node_attrs.memSyncDomainMapRemote;
        }
        if (metadata.node_attrs.has_preferred_shared_mem_carveout &&
            metadata.node_attrs.preferredSharedMemCarveout != 0) {
          kernel_node_attrs["preferredSharedMemCarveout"] =
              metadata.node_attrs.preferredSharedMemCarveout;
        }
        if (metadata.node_attrs.has_access_policy_window &&
            metadata.node_attrs.accessPolicyWindowNumBytes != 0) {
          kernel_node_attrs["accessPolicyWindowBasePtr"] = static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(metadata.node_attrs.accessPolicyWindowBasePtr));
          kernel_node_attrs["accessPolicyWindowNumBytes"] =
              metadata.node_attrs.accessPolicyWindowNumBytes;
          kernel_node_attrs["accessPolicyWindowHitRatio"] =
              metadata.node_attrs.accessPolicyWindowHitRatio;
          kernel_node_attrs["accessPolicyWindowHitProp"] =
              metadata.node_attrs.accessPolicyWindowHitProp;
          kernel_node_attrs["accessPolicyWindowMissProp"] =
              metadata.node_attrs.accessPolicyWindowMissProp;
        }
        if (metadata.node_attrs.has_device_updatable && metadata.node_attrs.deviceUpdatable != 0) {
          kernel_node_attrs["deviceUpdatable"] = metadata.node_attrs.deviceUpdatable;
          kernel_node_attrs["deviceUpdatableNode"] = static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(metadata.node_attrs.deviceUpdatableNode));
        }
        if (metadata.node_attrs.has_programmatic_stream_serialization &&
            metadata.node_attrs.programmaticStreamSerializationAllowed != 0) {
          kernel_node_attrs["programmaticStreamSerialization"] =
              metadata.node_attrs.programmaticStreamSerializationAllowed;
        }
        params["kernel_node_attrs"] = kernel_node_attrs;
      }

      json::array kernel_params_array;
      if (metadata.kernelParams) {
        for (int i = 0; i < metadata.num_params; ++i) {
          TORCH_CHECK(metadata.kernelParams[i], "kernelParams[", i, "] is null");
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);

          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.kernelParams[i]);
          for (size_t j = 0; j < std::get<1>(metadata.offset_and_sizes[i]); ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<int>(bytes[j]);
          }
          param_obj["value_hex"] = hex_stream.str();

          kernel_params_array.push_back(param_obj);
        }
      } else {
        for (int i = 0; i < metadata.num_params; ++i) {
          json::object param_obj;
          param_obj["index"] = i;
          param_obj["offset"] = std::get<0>(metadata.offset_and_sizes[i]);
          param_obj["size"] = std::get<1>(metadata.offset_and_sizes[i]);
          kernel_params_array.push_back(param_obj);
        }
      }
      params["kernelParams"] = kernel_params_array;

      if (metadata.extra) {
        void** config = static_cast<void**>(metadata.extra);
        json::array extra_array;
        size_t argBufferSize = 0;

        int idx = 0;
        while (config[idx] != CU_LAUNCH_PARAM_END) {
          if (config[idx] == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_POINTER");
            extra_array.push_back("null");
            idx += 2;
          } else if (config[idx] == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            extra_array.push_back("CU_LAUNCH_PARAM_BUFFER_SIZE");
            size_t* sizePtr = static_cast<size_t*>(config[idx + 1]);
            argBufferSize = *sizePtr;
            extra_array.push_back(static_cast<uint64_t>(argBufferSize));
            idx += 2;
          } else {
            extra_array.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(config[idx])));
            idx++;
          }
        }
        extra_array.push_back("CU_LAUNCH_PARAM_END");

        params["extra"] = extra_array;

        if (metadata.argBuffer && argBufferSize > 0) {
          std::ostringstream hex_stream;
          const auto* bytes = static_cast<const unsigned char*>(metadata.argBuffer);
          for (size_t j = 0; j < argBufferSize; ++j) {
            hex_stream << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<int>(bytes[j]);
          }
          params["extra_argBuffer_hex"] = hex_stream.str();
        } else {
          params["extra_argBuffer_hex"] = "";
        }
      } else {
        params["extra"] = json::array{};
        params["extra_argBuffer_hex"] = "";
      }

      CUkernel kern = metadata.kern;
      CUfunction func = metadata.func;
      std::string function_name;
      uint64_t binary_hash = 0;

      if (kern != nullptr) {
        const char* name = nullptr;
        if (cuKernelGetName(&name, kern) == CUDA_SUCCESS && name) {
          function_name = name;
        }

        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, kern) == CUDA_SUCCESS && lib != nullptr) {
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        }
      } else if (func != nullptr) {
        CUkernel temp_kern = reinterpret_cast<CUkernel>(func);
        CUlibrary lib = nullptr;
        if (cuKernelGetLibrary(&lib, temp_kern) == CUDA_SUCCESS && lib != nullptr) {
          const char* name = nullptr;
          if (cuKernelGetName(&name, temp_kern) == CUDA_SUCCESS && name) {
            function_name = name;
          }
          binary_hash = query_binary_hash(lib);
          mark_binary_used(binary_hash);
        } else {
          const char* name = nullptr;
          if (cuFuncGetName(&name, func) == CUDA_SUCCESS && name) {
            function_name = name;
          }

          CUmodule mod = nullptr;
          if (cuFuncGetModule(&mod, func) == CUDA_SUCCESS && mod != nullptr) {
            binary_hash = query_binary_hash(mod);
            mark_binary_used(binary_hash);
          }
        }
      }

      params["function_name"] = function_name;
      params["kernel_source_binary_hash"] = binary_hash;

      json::object func_attrs;

      // The function's dynamic-smem cap as it stands at capture. It is a
      // function attribute, not a node property: several kernels launch with
      // batch-dependent dynamic smem (FlashMLA's metadata kernel, DeepGEMM's
      // fp8 GEMMs), and recording this node's own sharedMemBytes here made
      // LOAD lower the shared cap under another graph's node.
      int max_dynamic_shared = 0;
      int preferred_carveout = 0;
      int cluster_scheduling = 0;
      int cluster_width = 0;
      int cluster_height = 0;
      int cluster_depth = 0;
      if (kern != nullptr) {
        cuKernelGetAttribute(&max_dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                             kern, capture_dev_);
        cuKernelGetAttribute(&preferred_carveout,
                             CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_scheduling,
                             CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, kern,
                             capture_dev_);
        cuKernelGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, kern,
                             capture_dev_);
      } else if (func != nullptr) {
        cuFuncGetAttribute(&max_dynamic_shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           func);
        cuFuncGetAttribute(&preferred_carveout, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                           func);
        cuFuncGetAttribute(&cluster_scheduling,
                           CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, func);
        cuFuncGetAttribute(&cluster_width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, func);
        cuFuncGetAttribute(&cluster_height, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT, func);
        cuFuncGetAttribute(&cluster_depth, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH, func);
      }
      // The cap cannot be below what this node launched with; a driver that
      // reports 0 for a function without an explicit opt-in is floored there.
      func_attrs["max_dynamic_shared_size_bytes"] =
          std::max(max_dynamic_shared, static_cast<int>(metadata.sharedMemBytes));
      func_attrs["preferred_shared_memory_carveout"] = preferred_carveout;
      func_attrs["cluster_scheduling_policy_preference"] = cluster_scheduling;
      // Save cluster dimensions - these are required when reconstructing the graph for kernels that
      // use clusters
      func_attrs["required_cluster_width"] = cluster_width;
      func_attrs["required_cluster_height"] = cluster_height;
      func_attrs["required_cluster_depth"] = cluster_depth;
      params["func_attrs"] = func_attrs;

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemcpyNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemcpyNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemcpyNode";

      json::object params;
      params["Depth"] = metadata.Depth;
      params["Height"] = metadata.Height;
      params["WidthInBytes"] = metadata.WidthInBytes;

      TORCH_CHECK(metadata.dstArray == nullptr, "dstArray must be null");
      TORCH_CHECK(metadata.srcArray == nullptr, "srcArray must be null");
      TORCH_CHECK(metadata.dstHost == nullptr, "dstHost must be null");
      TORCH_CHECK(metadata.srcHost == nullptr, "srcHost must be null");
      TORCH_CHECK(metadata.reserved0 == nullptr, "reserved0 must be null");
      TORCH_CHECK(metadata.reserved1 == nullptr, "reserved1 must be null");

      params["dstDevice"] = static_cast<uint64_t>(metadata.dstDevice);
      params["dstHeight"] = metadata.dstHeight;
      params["dstLOD"] = metadata.dstLOD;
      params["dstMemoryType"] = static_cast<int>(metadata.dstMemoryType);
      params["dstPitch"] = metadata.dstPitch;
      params["dstXInBytes"] = metadata.dstXInBytes;
      params["dstY"] = metadata.dstY;
      params["dstZ"] = metadata.dstZ;

      params["srcDevice"] = static_cast<uint64_t>(metadata.srcDevice);
      params["srcHeight"] = metadata.srcHeight;
      params["srcLOD"] = metadata.srcLOD;
      params["srcMemoryType"] = static_cast<int>(metadata.srcMemoryType);
      params["srcPitch"] = metadata.srcPitch;
      params["srcXInBytes"] = metadata.srcXInBytes;
      params["srcY"] = metadata.srcY;
      params["srcZ"] = metadata.srcZ;

      node_obj["params"] = params;

    } else if (std::holds_alternative<MemsetNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<MemsetNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "MemsetNode";

      json::object params;
      params["dst"] = static_cast<uint64_t>(metadata.dst);
      params["elementSize"] = metadata.elementSize;
      params["height"] = metadata.height;
      params["pitch"] = metadata.pitch;
      params["value"] = metadata.value;
      params["width"] = metadata.width;

      node_obj["params"] = params;

    } else if (std::holds_alternative<EventRecordNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventRecordNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventRecordNode";

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EventWaitNodeMetadata>(graphNode.metadata)) {
      const auto& metadata = std::get<EventWaitNodeMetadata>(graphNode.metadata);
      node_obj["type"] = "EventWaitNode";

      if (event_to_id.find(metadata.event) == event_to_id.end()) {
        event_to_id[metadata.event] = next_event_id++;
      }

      json::object params;
      params["event_id"] = event_to_id[metadata.event];
      node_obj["params"] = params;

    } else if (std::holds_alternative<EmptyNodeMetadata>(graphNode.metadata)) {
      node_obj["type"] = "EmptyNode";
      node_obj["params"] = json::object{};
    }

    nodes_array.push_back(node_obj);
  }

  // Compute topology key = node types + cluster dim values per kernel node.
  // cuGraphExecUpdate does NOT propagate kernel node attribute changes
  // (set via cuGraphKernelNodeSetAttribute) to the CUgraphExec — only
  // CUDA_KERNEL_NODE_PARAMS fields are updated. This means cluster dim
  // changes are silently ignored, causing "cluster misconfiguration" crashes
  // at launch when grid dims no longer match the stale cluster dims.
  // To avoid this, we include full cluster dim values in the topology key
  // so all graphs in a group share identical cluster dims.
  // This is computed BEFORE common attr extraction (which removes per-node attrs).
  {
    std::string topology_key;
    topology_key.reserve(nodes_array.size() * 16);
    for (size_t n = 0; n < nodes_array.size(); ++n) {
      if (n > 0)
        topology_key += ',';
      const json::object& no = nodes_array[n].as_object();
      std::string type = no.at("type").as_string().c_str();
      topology_key += type;
      if (type == "KernelNode") {
        unsigned int cdx = 0, cdy = 0, cdz = 0;
        const json::object& p = no.at("params").as_object();
        if (p.contains("kernel_node_attrs")) {
          const json::object& kna = p.at("kernel_node_attrs").as_object();
          if (kna.contains("clusterDimX")) {
            cdx = kna.at("clusterDimX").to_number<unsigned int>();
            cdy = kna.at("clusterDimY").to_number<unsigned int>();
            cdz = kna.at("clusterDimZ").to_number<unsigned int>();
          }
        }
        if (cdx == 0 && p.contains("func_attrs")) {
          const json::object& fa = p.at("func_attrs").as_object();
          if (fa.contains("required_cluster_width"))
            cdx = fa.at("required_cluster_width").to_number<unsigned int>();
          if (fa.contains("required_cluster_height"))
            cdy = fa.at("required_cluster_height").to_number<unsigned int>();
          if (fa.contains("required_cluster_depth"))
            cdz = fa.at("required_cluster_depth").to_number<unsigned int>();
        }
        if (cdx > 0 || cdy > 0 || cdz > 0) {
          // FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES=0: key only records *whether* a node launches
          // clusters, not the dims. Safe now that every member instantiates its own exec after
          // apply_on_demand_updates() sets its cluster dims per node (no cuGraphExecUpdate on the
          // path); it merges the groups that differ only by deep_gemm's per-M cluster size
          // (EP2: 26 -> ~3 templates). Default keeps the exact dims (previous behaviour).
          static const bool cluster_values_in_key = [] {
            const char* e = std::getenv("FOUNDRY_TOPOLOGY_KEY_CLUSTER_VALUES");
            return !(e && (std::string(e) == "0" || std::string(e) == "false"));
          }();
          if (cluster_values_in_key) {
            topology_key +=
                ":C" + std::to_string(cdx) + "_" + std::to_string(cdy) + "_" + std::to_string(cdz);
          } else {
            topology_key += ":C";
          }
        } else {
          topology_key += ":0";
        }
      }
    }
    // The edges too. LOAD rebuilds a member on its template's graph, so the member runs with the
    // template's edges (its own are not stored). Kernels that use programmatic dependent launch
    // for some batch sizes only (FA3 prepare -> fwd on Hopper) give graphs with the same nodes
    // and a programmatic edge in one size, a full one in another; a member that inherits a
    // programmatic edge it did not have starts before its producer is done.
    {
      std::vector<std::array<uint32_t, 5>> edges;
      edges.reserve(graph_dependencies.size());
      for (const auto& dep : graph_dependencies)
        edges.push_back({static_cast<uint32_t>(dep.from_index), static_cast<uint32_t>(dep.to_index),
                         dep.from_port, dep.to_port, dep.edge_type});
      std::sort(edges.begin(), edges.end());
      uint64_t h = 1469598103934665603ull;  // FNV-1a over the sorted edge list
      for (const auto& e : edges)
        for (uint32_t v : e)
          for (int b = 0; b < 4; ++b) {
            h ^= (v >> (8 * b)) & 0xffu;
            h *= 1099511628211ull;
          }
      std::ostringstream edge_key;
      edge_key << "|E" << edges.size() << ':' << std::hex << std::setw(16) << std::setfill('0')
               << h;
      topology_key += edge_key.str();
    }
    root["topology_key"] = topology_key;
  }

  // Extract common kernel node attributes shared by ALL kernel nodes.
  // If all kernel nodes have the same non-default attrs, store them once at
  // the graph level and remove from individual nodes to avoid redundant
  // cuGraphKernelNodeSetAttribute calls on load.
  {
    bool first_kernel = true;
    json::object common_attrs;
    bool all_same = true;

    for (const auto& node_val : nodes_array) {
      const json::object& node_obj_ref = node_val.as_object();
      if (node_obj_ref.at("type").as_string() != "KernelNode")
        continue;
      const json::object& params_ref = node_obj_ref.at("params").as_object();
      if (!params_ref.contains("kernel_node_attrs"))
        continue;
      const json::object& attrs = params_ref.at("kernel_node_attrs").as_object();

      // Build a comparable version (exclude attrQueryAvailable which is metadata)
      json::object settable_attrs;
      for (const auto& kv : attrs) {
        if (kv.key() != "attrQueryAvailable") {
          settable_attrs[kv.key()] = kv.value();
        }
      }

      if (first_kernel) {
        common_attrs = settable_attrs;
        first_kernel = false;
      } else if (json::serialize(settable_attrs) != json::serialize(common_attrs)) {
        all_same = false;
        break;
      }
    }

    if (!first_kernel && all_same && !common_attrs.empty()) {
      root["common_kernel_node_attrs"] = common_attrs;
      // Remove per-node attrs that are now in the common section
      for (auto& node_val : nodes_array) {
        json::object& node_obj_ref = node_val.as_object();
        if (node_obj_ref.at("type").as_string() != "KernelNode")
          continue;
        json::object& params_ref = node_obj_ref.at("params").as_object();
        if (!params_ref.contains("kernel_node_attrs"))
          continue;
        // Keep only attrQueryAvailable in per-node attrs
        json::object trimmed;
        const json::object& attrs = params_ref.at("kernel_node_attrs").as_object();
        if (attrs.contains("attrQueryAvailable")) {
          trimmed["attrQueryAvailable"] = attrs.at("attrQueryAvailable");
        }
        params_ref["kernel_node_attrs"] = trimmed;
      }
    }
  }

  root["nodes"] = nodes_array;

  json::array deps_array;
  for (const auto& dep : graph_dependencies) {
    json::object dep_obj;
    dep_obj["from"] = dep.from_index;
    dep_obj["to"] = dep.to_index;
    // Edge data (ports/type) only when non-default — default edges stay
    // compact and load exactly as before.
    if (dep.from_port != 0) {
      dep_obj["from_port"] = dep.from_port;
    }
    if (dep.to_port != 0) {
      dep_obj["to_port"] = dep.to_port;
    }
    if (dep.edge_type != 0) {
      dep_obj["edge_type"] = dep.edge_type;
    }
    deps_array.push_back(dep_obj);
  }
  root["dependencies"] = deps_array;

  std::vector<std::tuple<uint64_t, uint64_t, uint64_t, uintptr_t, uintptr_t>> sorted_generators;
  for (const auto& [state, wholegraph_increment] : captured_generator_states_) {
    uint64_t state_id = global_generator_state_registry.query_state_id(state.get());
    auto [seed_ptr, offset_ptr] = wholegraph_increment > 0 ? rng_state_addresses(state)
                                                           : std::pair<uintptr_t, uintptr_t>{0, 0};
    sorted_generators.emplace_back(state_id, state->seed_, wholegraph_increment, seed_ptr,
                                   offset_ptr);
  }

  std::sort(sorted_generators.begin(), sorted_generators.end(),
            [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

  json::array generators_array;
  for (const auto& [state_id, seed, wholegraph_increment, seed_ptr, offset_ptr] :
       sorted_generators) {
    json::object gen_obj;
    gen_obj["id"] = state_id;
    gen_obj["seed"] = seed;
    gen_obj["wholegraph_increment"] = wholegraph_increment;
    if (seed_ptr != 0 || offset_ptr != 0) {
      // Where this graph's Philox kernels read the seed and offset (LOAD checks it).
      gen_obj["seed_ptr"] = static_cast<uint64_t>(seed_ptr);
      gen_obj["offset_ptr"] = static_cast<uint64_t>(offset_ptr);
    }
    generators_array.push_back(gen_obj);
  }
  root["generators"] = generators_array;

  root["allocator_events"] = allocator_events_;

  json::object output_tensors_obj;
  output_tensors_obj["type"] = static_cast<int>(output_type);

  json::array tensors_array;
  if (std::holds_alternative<at::Tensor>(output_tensors)) {
    tensors_array.push_back(serialize_tensor_metadata(std::get<at::Tensor>(output_tensors)));
  } else if (std::holds_alternative<std::vector<at::Tensor>>(output_tensors)) {
    for (const auto& t : std::get<std::vector<at::Tensor>>(output_tensors)) {
      tensors_array.push_back(serialize_tensor_metadata(t));
    }
  }
  output_tensors_obj["tensors"] = tensors_array;
  root["output_tensors"] = output_tensors_obj;

  // The binary format for fast loading sits next to the JSON.
  std::string bin_path = json_path;
  if (bin_path.size() >= 5 && bin_path.substr(bin_path.size() - 5) == ".json") {
    bin_path = bin_path.substr(0, bin_path.size() - 5) + ".cugraph";
  } else {
    bin_path += ".cugraph";
  }
  // LOAD prefers the binary: one left by an earlier SAVE must not outlive the new JSON when
  // this SAVE stops before its own binary is written.
  std::remove(bin_path.c_str());

  const std::string json_tmp = json_path + ".tmp";
  std::ofstream file(json_tmp, std::ios::trunc);
  TORCH_CHECK(file.is_open(), "Failed to open file for writing: ", json_tmp);
  file << json::serialize(root);
  file.close();
  commit_file(json_tmp, json_path, !file.fail());

  save_binary(bin_path, root);
}

// save_binary is implemented in BinaryGraphIO.cpp

// One graph from its JSON: its allocations are replayed first (the node params
// may point at them), then it is built by the template builder of the split
// loader (common kernel attributes, 'extra' and parameter checks).
GraphLoadResult CUDAGraph::load(const std::string& json_path, MempoolId_t pool) {
  std::ifstream file(json_path);
  TORCH_CHECK(file.is_open(), "Failed to open file for reading: ", json_path);
  std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  boost::json::value root_val = boost::json::parse(json_str);
  boost::json::object& root = root_val.as_object();
  // Checked before anything below replays allocations or moves the cursor.
  for (const char* key : {"generators", "allocator_events", "nodes", "dependencies"}) {
    TORCH_CHECK(root.contains(key), json_path, " has no \"", key, "\"",
                std::string(key) == "dependencies"
                    ? " (a member of a topology group: load it with start_graph_builds "
                      "together with its template)"
                    : "");
  }
  boost::json::array generators = std::move(root.at("generators").as_array());
  root["generators"] = boost::json::array{};
  boost::json::object events = root.at("allocator_events").as_object();

  ParsedGraphData parsed =
      prepare_graph_shell(std::move(root_val), pool, global_generator_state_registry);
  restore_graph_state(*parsed.graph, generators, events, &global_generator_state_registry, 0);
  CUcontext ctx = nullptr;
  C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&ctx));
  return build_graph_from_parsed(std::move(parsed), ctx, reconstruct_tensor_from_metadata, nullptr);
}

std::shared_ptr<PendingGraphLoads> CUDAGraph::start_graph_builds(
    const std::vector<std::string>& json_paths, MempoolId_t pool, int num_threads) {
  return start_graph_builds_impl(json_paths, pool, num_threads, global_generator_state_registry);
}

std::vector<GraphLoadResult> CUDAGraph::finish_graph_loads(
    std::shared_ptr<PendingGraphLoads> pending) {
  return finish_graph_loads_impl(std::move(pending), reconstruct_tensor_from_metadata);
}

GraphLoadResult CUDAGraph::finish_one_graph_load(std::shared_ptr<PendingGraphLoads> pending,
                                                 size_t index) {
  return finish_one_graph_load_impl(std::move(pending), index, reconstruct_tensor_from_metadata);
}

// ---- Kernel function attributes (contract in CUDAGraphInternal.h) ----------

namespace {

std::mutex g_smem_cap_mutex;
// CUkernel / CUfunction handle -> MAX_DYNAMIC_SHARED_SIZE_BYTES this process set on it.
std::unordered_map<const void*, int> g_smem_cap_high_water;

// Raise one handle's cap to `need` unless it is already there. Caller holds
// g_smem_cap_mutex. A failed set is reported and leaves the mark unchanged so
// the next node with the same need retries.
template <typename SetFn>
void raise_smem_cap_locked(const void* key, int need, SetFn&& set, const char* where) {
  int& cap = g_smem_cap_high_water[key];
  if (cap >= need) {
    return;
  }
  CUresult r = set();
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "[foundry %s] setting MAX_DYNAMIC_SHARED_SIZE_BYTES=%d failed: %d\n", where,
            need, (int)r);
    return;
  }
  cap = need;
}

void raise_dynamic_smem_cap(CUkernel kern, CUfunction func, CUdevice dev, int need,
                            const char* where) {
  if (need <= 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_smem_cap_mutex);
  if (kern != nullptr) {
    raise_smem_cap_locked(
        kern, need,
        [&] {
          return cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need, kern,
                                      dev);
        },
        where);
    CUfunction ctx_func = nullptr;
    CUresult r = cuKernelGetFunction(&ctx_func, kern);
    if (r != CUDA_SUCCESS || ctx_func == nullptr) {
      fprintf(stderr,
              "[foundry %s] cuKernelGetFunction failed (%d): dynamic smem cap %d not applied to "
              "the per-context function\n",
              where, (int)r, need);
    } else {
      raise_smem_cap_locked(
          ctx_func, need,
          [&] {
            return cuFuncSetAttribute(ctx_func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                      need);
          },
          where);
    }
  }
  if (func != nullptr) {
    raise_smem_cap_locked(
        func, need,
        [&] {
          return cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, need);
        },
        where);
  }
}

}  // namespace

const std::vector<KernelParamInfo>* declared_kernel_params(
    const std::variant<CUfunction, CUkernel>& handle) {
  static std::shared_mutex mu;
  static std::unordered_map<const void*, std::optional<std::vector<KernelParamInfo>>> cache;
  const void* key = std::holds_alternative<CUkernel>(handle)
                        ? static_cast<const void*>(std::get<CUkernel>(handle))
                        : static_cast<const void*>(std::get<CUfunction>(handle));
  {
    std::shared_lock<std::shared_mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second ? &*it->second : nullptr;
  }
  std::optional<std::vector<KernelParamInfo>> params(std::in_place);
  for (size_t i = 0;; ++i) {
    size_t offset = 0, size = 0;
    CUresult r = std::holds_alternative<CUkernel>(handle)
                     ? cuKernelGetParamInfo(std::get<CUkernel>(handle), i, &offset, &size)
                     : cuFuncGetParamInfo(std::get<CUfunction>(handle), i, &offset, &size);
    if (r != CUDA_SUCCESS) {
      // Past the last parameter the driver returns CUDA_ERROR_INVALID_VALUE; any other
      // error means the declaration is not available (nothing is checked then).
      if (r != CUDA_ERROR_INVALID_VALUE)
        params.reset();
      break;
    }
    params->push_back({offset, size});
  }
  std::unique_lock<std::shared_mutex> lock(mu);
  auto& entry = cache.emplace(key, std::move(params)).first->second;
  return entry ? &*entry : nullptr;
}

void check_archived_kernel_params(const std::variant<CUfunction, CUkernel>& handle,
                                  const std::string& function_name, size_t num_params,
                                  const std::function<size_t(size_t)>& size_of) {
  const auto* declared = declared_kernel_params(handle);
  if (declared == nullptr)
    return;
  TORCH_CHECK(num_params == declared->size(), "archived kernel ", function_name, " has ",
              num_params, " parameters, the loaded function declares ", declared->size());
  for (size_t i = 0; i < num_params; ++i)
    TORCH_CHECK(size_of(i) == (*declared)[i].size, "archived kernel ", function_name,
                ": parameter ", i, " has ", size_of(i), " bytes, the loaded function declares ",
                (*declared)[i].size);
}

void apply_saved_function_attributes(const std::variant<CUfunction, CUkernel>& handle, CUdevice dev,
                                     int max_dynamic_shared_size_bytes,
                                     int preferred_shared_memory_carveout, const char* where) {
  // CLUSTER_SCHEDULING_POLICY_PREFERENCE is recorded but not reapplied: it is
  // only a hint and fails with cudaErrorInvalidClusterSize when the kernel's
  // compiled cluster requirements differ from the saved preference.
  if (std::holds_alternative<CUkernel>(handle)) {
    CUkernel kern = std::get<CUkernel>(handle);
    raise_dynamic_smem_cap(kern, nullptr, dev, max_dynamic_shared_size_bytes, where);
    if (preferred_shared_memory_carveout >= 0) {
      C10_CUDA_DRIVER_CHECK(cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                                 preferred_shared_memory_carveout, kern, dev));
    }
  } else {
    CUfunction func = std::get<CUfunction>(handle);
    raise_dynamic_smem_cap(nullptr, func, dev, max_dynamic_shared_size_bytes, where);
    if (preferred_shared_memory_carveout >= 0) {
      C10_CUDA_DRIVER_CHECK(cuFuncSetAttribute(func,
                                               CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                               preferred_shared_memory_carveout));
    }
  }
}

void ensure_dynamic_smem_optin(const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev,
                               const char* where) {
  raise_dynamic_smem_cap(params.kern, params.func, dev, static_cast<int>(params.sharedMemBytes),
                         where);
}

CUresult add_restored_kernel_node(CUgraphNode* node, CUgraph graph,
                                  const CUDA_KERNEL_NODE_PARAMS& params, CUdevice dev, unsigned cx,
                                  unsigned cy, unsigned cz, const char* where) {
  ensure_dynamic_smem_optin(params, dev, where);
  return add_kernel_node_cluster_optin(node, graph, params, dev, cx, cy, cz, where);
}

}  // namespace foundry
