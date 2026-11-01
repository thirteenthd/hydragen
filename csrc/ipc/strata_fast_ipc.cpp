// SPDX-License-Identifier: Apache-2.0
// Implementation of high-performance C++ SPSC IPC reader for Strata engine stdout.

#include "strata_fast_ipc.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#else
#include <io.h>
#endif

namespace strata::ipc {

struct TokenPacket {
    int32_t token_id;
    int32_t status; // 1 = token, -1 = done, -2 = error
};

template <size_t Capacity = 4096>
class SpscRingBuffer {
public:
    SpscRingBuffer() : m_head(0), m_tail(0) {}

    bool push(const TokenPacket& pkt) {
        size_t h = m_head.load(std::memory_order_relaxed);
        size_t next_h = (h + 1) % Capacity;
        if (next_h == m_tail.load(std::memory_order_acquire)) {
            return false; // Full
        }
        m_buffer[h] = pkt;
        m_head.store(next_h, std::memory_order_release);
        return true;
    }

    bool pop(TokenPacket& pkt) {
        size_t t = m_tail.load(std::memory_order_relaxed);
        if (t == m_head.load(std::memory_order_acquire)) {
            return false; // Empty
        }
        pkt = m_buffer[t];
        m_tail.store((t + 1) % Capacity, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return m_tail.load(std::memory_order_relaxed) == m_head.load(std::memory_order_acquire);
    }

private:
    TokenPacket m_buffer[Capacity];
    alignas(64) std::atomic<size_t> m_head;
    alignas(64) std::atomic<size_t> m_tail;
};

struct SlotContext {
    SpscRingBuffer<4096> ring;
    std::mutex mtx;
    std::condition_variable cv;
};

class FastEngineIpc {
public:
    FastEngineIpc(int stdout_fd, int num_slots)
        : m_fd(stdout_fd), m_num_slots(std::max(1, num_slots)), m_running(true),
          m_control_cb(nullptr), m_control_user_data(nullptr) {
        for (int i = 0; i < m_num_slots; ++i) {
            m_slots.push_back(std::make_unique<SlotContext>());
        }
        m_reader_thread = std::thread(&FastEngineIpc::reader_loop, this);
    }

    ~FastEngineIpc() {
        stop();
        if (m_reader_thread.joinable()) {
            m_reader_thread.join();
        }
    }

    void set_control_callback(strata_control_callback_fn cb, void* user_data) {
        std::lock_guard<std::mutex> lock(m_cb_mutex);
        m_control_cb = cb;
        m_control_user_data = user_data;
    }

    void stop() {
        if (!m_running.exchange(false)) return;
        // Wake up any waiting consumers
        for (int i = 0; i < m_num_slots; ++i) {
            std::lock_guard<std::mutex> lock(m_slots[i]->mtx);
            m_slots[i]->cv.notify_all();
        }
    }

    int32_t read_token(int32_t slot, int32_t* out_token, int32_t timeout_ms) {
        if (slot < 0 || slot >= m_num_slots) return -2;

        SlotContext& sc = *m_slots[slot];
        TokenPacket pkt{0, 0};

        // 1. Fast Lock-Free path (< 50 nanoseconds)
        if (sc.ring.pop(pkt)) {
            if (pkt.status == 1) {
                *out_token = pkt.token_id;
                return 1;
            }
            return pkt.status;
        }

        if (timeout_ms == 0) return 0; // Immediate non-blocking poll

        // 2. Wait on condition variable if empty
        std::unique_lock<std::mutex> lock(sc.mtx);
        auto pred = [this, &sc, &pkt]() {
            return sc.ring.pop(pkt) || !m_running.load(std::memory_order_relaxed);
        };

        if (timeout_ms < 0) {
            sc.cv.wait(lock, pred);
        } else {
            if (!sc.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred)) {
                return 0; // Timeout
            }
        }

        if (pkt.status == 1) {
            *out_token = pkt.token_id;
            return 1;
        }
        if (pkt.status != 0) {
            return pkt.status;
        }
        return m_running.load() ? 0 : -2;
    }

private:
    void notify_slot(int slot) {
        if (slot >= 0 && slot < m_num_slots) {
            std::lock_guard<std::mutex> lock(m_slots[slot]->mtx);
            m_slots[slot]->cv.notify_one();
        }
    }

    void broadcast_all(int status) {
        for (int i = 0; i < m_num_slots; ++i) {
            TokenPacket pkt{0, status};
            m_slots[i]->ring.push(pkt);
            notify_slot(i);
        }
    }

    void forward_control(const char* line) {
        strata_control_callback_fn cb = nullptr;
        void* udata = nullptr;
        {
            std::lock_guard<std::mutex> lock(m_cb_mutex);
            cb = m_control_cb;
            udata = m_control_user_data;
        }
        if (cb != nullptr) {
            cb(line, udata);
        }
    }

    void parse_line(const char* line, size_t len) {
        if (len == 0) return;

        // Fast parsing of "BT <slot> <token_id>" -> Zero Python GIL, Zero String Allocation
        if (len >= 5 && line[0] == 'B' && line[1] == 'T' && line[2] == ' ') {
            const char* p = line + 3;
            int slot = 0;
            while (*p >= '0' && *p <= '9') {
                slot = slot * 10 + (*p - '0');
                p++;
            }
            while (*p == ' ') p++;
            int32_t tok = 0;
            while (*p >= '0' && *p <= '9') {
                tok = tok * 10 + (*p - '0');
                p++;
            }
            if (slot >= 0 && slot < m_num_slots) {
                m_slots[slot]->ring.push({tok, 1});
                notify_slot(slot);
            }
            return;
        }

        // Fast parsing of "BDONE <slot>"
        if (len >= 6 && std::strncmp(line, "BDONE ", 6) == 0) {
            const char* p = line + 6;
            int slot = 0;
            while (*p >= '0' && *p <= '9') {
                slot = slot * 10 + (*p - '0');
                p++;
            }
            if (slot >= 0 && slot < m_num_slots) {
                m_slots[slot]->ring.push({0, -1});
                notify_slot(slot);
            }
            forward_control(line);
            return;
        }

        // Fast parsing of single-stream "T <token_id>"
        if (len >= 3 && line[0] == 'T' && line[1] == ' ') {
            const char* p = line + 2;
            int32_t tok = 0;
            while (*p >= '0' && *p <= '9') {
                tok = tok * 10 + (*p - '0');
                p++;
            }
            m_slots[0]->ring.push({tok, 1});
            notify_slot(0);
            forward_control(line);
            return;
        }

        // Fast parsing of single-stream "DONE"
        if (len >= 4 && std::strncmp(line, "DONE", 4) == 0) {
            m_slots[0]->ring.push({0, -1});
            notify_slot(0);
            forward_control(line);
            return;
        }

        // Error lines
        if (len >= 3 && std::strncmp(line, "ERR", 3) == 0) {
            broadcast_all(-2);
        }

        // Forward all other control lines (BSTAT, PSTAT, INFO, PP, FATAL, etc.)
        forward_control(line);
    }

    void reader_loop() {
        constexpr size_t BUF_SIZE = 65536;
        std::vector<char> raw_buf(BUF_SIZE);
        std::string line_accum;
        line_accum.reserve(1024);

        while (m_running.load(std::memory_order_relaxed)) {
#if defined(__unix__) || defined(__APPLE__)
            ssize_t n = ::read(m_fd, raw_buf.data(), BUF_SIZE);
#else
            int n = ::_read(m_fd, raw_buf.data(), static_cast<unsigned int>(BUF_SIZE));
#endif
            if (n <= 0) {
                // EOF or error
                broadcast_all(-2);
                forward_control("EOF");
                break;
            }

            for (ssize_t i = 0; i < n; ++i) {
                char c = raw_buf[i];
                if (c == '\n') {
                    if (!line_accum.empty() && line_accum.back() == '\r') {
                        line_accum.pop_back();
                    }
                    parse_line(line_accum.data(), line_accum.size());
                    line_accum.clear();
                } else {
                    line_accum.push_back(c);
                }
            }
        }
    }

    int m_fd;
    int m_num_slots;
    std::atomic<bool> m_running;
    std::thread m_reader_thread;

    std::mutex m_cb_mutex;
    strata_control_callback_fn m_control_cb;
    void* m_control_user_data;

    std::vector<std::unique_ptr<SlotContext>> m_slots;
};

} // namespace strata::ipc

extern "C" {

strata_fast_ipc_t* strata_fast_ipc_create(int stdout_fd, int num_slots) {
    try {
        return reinterpret_cast<strata_fast_ipc_t*>(new strata::ipc::FastEngineIpc(stdout_fd, num_slots));
    } catch (...) {
        return nullptr;
    }
}

void strata_fast_ipc_set_control_callback(strata_fast_ipc_t* ipc, strata_control_callback_fn cb, void* user_data) {
    if (ipc) {
        reinterpret_cast<strata::ipc::FastEngineIpc*>(ipc)->set_control_callback(cb, user_data);
    }
}

int32_t strata_fast_ipc_read_token(strata_fast_ipc_t* ipc, int32_t slot, int32_t* out_token, int32_t timeout_ms) {
    if (!ipc) return -2;
    return reinterpret_cast<strata::ipc::FastEngineIpc*>(ipc)->read_token(slot, out_token, timeout_ms);
}

void strata_fast_ipc_stop(strata_fast_ipc_t* ipc) {
    if (ipc) {
        reinterpret_cast<strata::ipc::FastEngineIpc*>(ipc)->stop();
    }
}

void strata_fast_ipc_destroy(strata_fast_ipc_t* ipc) {
    if (ipc) {
        delete reinterpret_cast<strata::ipc::FastEngineIpc*>(ipc);
    }
}

}
