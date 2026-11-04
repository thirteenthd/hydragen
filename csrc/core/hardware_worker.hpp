// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace hydragen::core {

/**
 * @brief Abstract interface for Hardware Execution Workers.
 * Isolates compute execution (e.g. bin/strata, native tensorrt-llm, or shared lib)
 * from the high-level HydragenEngine scheduling, locality routing, and Paged KV management.
 */
class IHardwareWorker {
public:
    virtual ~IHardwareWorker() = default;
    virtual bool initialize(const std::string& exe_path, const std::vector<std::string>& args) = 0;
    virtual int32_t step_token(int32_t slot_id, int32_t token_id) = 0;
    virtual void close() = 0;
    virtual bool is_alive() const = 0;
    virtual const char* get_worker_type() const = 0;
};

/**
 * @brief Lightweight MockHardwareWorker for deterministic offline execution & unit tests.
 */
class MockHardwareWorker : public IHardwareWorker {
public:
    MockHardwareWorker() : alive_(true) {}
    ~MockHardwareWorker() override = default;

    bool initialize(const std::string& /*exe_path*/, const std::vector<std::string>& /*args*/) override {
        alive_ = true;
        return true;
    }

    int32_t step_token(int32_t /*slot_id*/, int32_t token_id) override {
        return token_id + 1;
    }

    void close() override {
        alive_ = false;
    }

    bool is_alive() const override {
        return alive_;
    }

    const char* get_worker_type() const override {
        return "mock_hardware_worker";
    }

private:
    bool alive_;
};

} // namespace hydragen::core
