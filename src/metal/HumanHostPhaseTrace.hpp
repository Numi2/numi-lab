#pragma once
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>

namespace metalrobo {
// Host-only diagnostic intervals. Marks do not time GPU execution and nested
// traces include their diagnostic overhead. Construct before the owner's lock
// scope so output follows its own lock release. Outer callers may still hold
// their own locks; nested diagnostic output is part of their host interval.
class HumanHostPhaseTrace final {
 public:
  HumanHostPhaseTrace(bool standing, std::uint32_t step, const char* prefix,
      bool sampleEvery1024 = false)
      : step_(step), prefix_(prefix) {
    const char* setting = std::getenv("NUMI_HUMAN_HOST_INNER_PROFILE");
    enabled_ = standing && (step < 8u ||
        (sampleEvery1024 && step % 1024u == 0u)) && setting != nullptr &&
        std::strcmp(setting, "1") == 0;
    if (enabled_) last_ = Clock::now();
  }
  HumanHostPhaseTrace(const HumanHostPhaseTrace&) = delete;
  HumanHostPhaseTrace& operator=(const HumanHostPhaseTrace&) = delete;
  void mark(const char* stage) noexcept {
    if (!enabled_ || count_ == records_.size()) return;
    const auto now = Clock::now();
    records_[count_++] = {stage, static_cast<long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_).count())};
    last_ = now;
  }
  void transaction(const void* callback, std::uint32_t phase,
      bool valid, std::uint32_t step) noexcept {
    if (!enabled_ || callbackCount_ == callbacks_.size()) return;
    callbacks_[callbackCount_++] = {callback, phase, valid, step};
  }
  ~HumanHostPhaseTrace() noexcept {
    for (std::size_t i = 0u; i < callbackCount_; ++i) {
      Dl_info info{};
      const auto& entry = callbacks_[i];
      const bool found = entry.callback != nullptr && dladdr(entry.callback, &info);
      std::fprintf(stderr, "human_tx_phase=%u valid=%u step=%u image=%s symbol=%s\n",
          entry.phase, static_cast<unsigned>(entry.valid), entry.step,
          found && info.dli_fname ? info.dli_fname : "none",
          found && info.dli_sname ? info.dli_sname : "none");
    }
    for (std::size_t i = 0u; i < count_; ++i)
      std::fprintf(stderr, "%s=%s step=%u elapsed_ns=%lld\n", prefix_,
          records_[i].stage, step_, records_[i].elapsed);
  }
 private:
  using Clock = std::chrono::steady_clock;
  struct Record { const char* stage = nullptr; long long elapsed = 0; };
  struct Callback { const void* callback = nullptr; std::uint32_t phase = 0;
    bool valid = false; std::uint32_t step = 0; };
  std::array<Callback, 8u> callbacks_{};
  std::size_t callbackCount_ = 0u;
  std::array<Record, 32u> records_{};
  std::size_t count_ = 0u;
  std::uint32_t step_;
  const char* prefix_;
  bool enabled_ = false;
  Clock::time_point last_{};
};
} // namespace metalrobo
