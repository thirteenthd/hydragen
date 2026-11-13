#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage. Appending never relocates existing payloads;
// only the small segment directory can move. No sharing or copy-on-write is
// needed because a parked image and its retained active buffers have one owner.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;
    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    // A copy never aliases: each segment's bytes go to a fresh store. (Adoption shares, below.)
    ConversationBuffer(const ConversationBuffer& other) : size_(other.size_) {
        segments_.reserve(other.segments_.size());
        for (const auto& s : other.segments_) {
            Segment c;
            auto store = std::make_shared<Seg>();
            store->assign(s.store->data() + s.off, s.store->data() + s.off + s.len);
            c.store = std::move(store);
            c.cap = c.store->capacity();
            c.len = c.store->size();
            segments_.push_back(std::move(c));
        }
    }
    ConversationBuffer& operator=(const ConversationBuffer& other) {
        if (this != &other) *this = ConversationBuffer(other);
        return *this;
    }
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : segments_(std::move(other.segments_)), size_(std::exchange(other.size_, 0)) {}
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        segments_ = std::move(other.segments_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t bytes() const {
        size_t n = segments_.capacity() * sizeof(Segment);
        for (const auto& segment : segments_) n += segment.capacity();
        return n;
    }
    // Includes the old directory while reserve allocates its replacement.
    // Saturation makes an overflowing estimate fail admission before resize.
    size_t allocation_peak(size_t n) const {
        size_t total = bytes();
        if (n <= size_) return total;
        const size_t spare = segments_.empty() ? 0 : (segments_.back().capacity() > segments_.back().size() ? segments_.back().capacity() - segments_.back().size() : 0);
        const size_t extra = n - size_ - std::min(n - size_, spare);
        const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
        if (!add(total, full * segment_bytes)) return SIZE_MAX;
        if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                segments_.empty() ? 0 : segments_.back().capacity(), size_ != 0, size_))) return SIZE_MAX;
        size_t count = segments_.size();
        if (!add(count, full + (tail != 0))) return SIZE_MAX;
        if (count > segments_.capacity() &&
            (count > SIZE_MAX / sizeof(Segment) || !add(total, count * sizeof(Segment)))) return SIZE_MAX;
        return total;
    }
    void resize(size_t n, uint8_t value = 0) { resize_impl(n, &value); }
    void resize_impl(size_t n, const uint8_t* value) {
        if (n > size_) {
            const size_t held = size_;
            const bool growing = held != 0;
            const size_t spare = segments_.empty() ? 0 : (segments_.back().capacity() > segments_.back().size() ? segments_.back().capacity() - segments_.back().size() : 0);
            const size_t extend = std::min(n - size_, spare);
            const size_t extra = n - size_ - extend;
            segments_.reserve(segments_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
            if (extend) {
                auto& back = segments_.back();
                if (value) back.store->resize(back.store->size() + extend, *value);
                else back.store->resize(back.store->size() + extend);
                back.len += extend;
                back.cap = back.store->capacity();
                size_ += extend;
            }
            while (size_ < n) {
                const size_t count = std::min(segment_bytes, n - size_);
                const size_t previous = segments_.empty() ? 0 : segments_.back().capacity();
                auto store = std::make_shared<Seg>();
                store->reserve(segment_capacity(count, previous, growing, held));
                if (value) store->resize(count, *value);
                else store->resize(count);
                Segment segment;
                segment.store = std::move(store);
                segment.cap = segment.store->capacity();
                segment.len = count;
                segments_.push_back(std::move(segment));
                size_ += count;
                progress_beat();   // #2049: zero-filling 16 MiB of fresh pages is slow when the host is paging
            }
        } else {
            while (!segments_.empty() && size_ - segments_.back().size() >= n) {
                size_ -= segments_.back().size();
                segments_.pop_back();
            }
            if (size_ > n) {
                auto& back = segments_.back();
                if (back.cap) {   // owned (cap>0): give the bytes back
                    back.store->resize(back.store->size() - (size_ - n));
                    back.cap = back.store->capacity();
                }
                back.len -= size_ - n;
            }
            size_ = n;
        }
    }
    // M-C2 host-tier CoW: replace this buffer with VIEWS of src's leading `n` bytes (no allocation, no
    // copy) - the caller then appends its own bytes past n. A view shares src's store; it counts 0 in
    // bytes()/allocation, and any write that would land in a borrowed segment fails (see visit).
    // Verify-then-view: if `mine` is given, the adopted range must be byte-identical to it BEFORE
    // anything mutates (siblings that prefilled a shared head under different chunk plans hold
    // different bytes); a mismatch leaves this buffer untouched and the caller copies instead.
    bool adopt_shared(const ConversationBuffer& src, size_t n, const uint8_t* mine = nullptr) {
        if (n > src.size_) return false;
        std::vector<std::pair<const Segment*, size_t>> plan;
        size_t at = 0, pos = 0;
        for (const auto& s : src.segments_) {
            if (at >= n) break;
            const size_t take = std::min(s.len, n - at);
            if (take) {
                if (mine && (s.store->size() < s.off + take ||
                             std::memcmp(mine + pos, s.store->data() + s.off, take) != 0))
                    return false;
                plan.push_back({&s, take});
                pos += take;
                at += take;
            }
        }
        if (at != n) return false;
        segments_.clear(); size_ = 0;
        for (const auto& [s, take] : plan) {
            Segment v;
            v.store = s->store;
            v.off = s->off;
            v.len = take;
            v.cap = 0;
            segments_.push_back(std::move(v));
            size_ += take;
        }
        return true;
    }

    void pop_back() { resize(size_ - 1); }
    // resize() without the zero fill of what it adds (the fresh pages are not touched here): only for a caller that
    // writes every added byte before anything reads it (STRATA_PARK_ASYNC's background copy)
    void resize_for_overwrite(size_t n) { resize_impl(n, nullptr); }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t a = 0, b = 0, x = 0, y = 0;
        while (a < segments_.size() && b < other.segments_.size()) {
            const auto& left = segments_[a];
            const auto& right = other.segments_[b];
            const size_t n = std::min(left.size() - x, right.size() - y);
            if (std::memcmp(left.data() + x, right.data() + y, n)) return false;
            x += n; y += n;
            if (x == left.size()) { ++a; x = 0; }
            if (y == right.size()) { ++b; y = 0; }
        }
        return true;
    }

private:
    // value-initializes only when asked (resize's explicit fill value): resize_for_overwrite leaves the bytes alone
    template<class T> struct DefaultInit : std::allocator<T> {
        template<class U> struct rebind { using other = DefaultInit<U>; };
        DefaultInit() = default;
        template<class U> DefaultInit(const DefaultInit<U>&) noexcept {}
        template<class U> void construct(U* p) { ::new ((void*) p) U; }
        template<class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
    };
    using Seg = std::vector<uint8_t, DefaultInit<uint8_t>>;
    // A run of bytes: a window [off, off+len) of `store`. cap>0 iff this buffer owns the store outright
    // and off==0 (cap: the store's allocated capacity); a borrowed view carries cap==0 and counts 0.
    struct Segment {
        std::shared_ptr<Seg> store;
        size_t off = 0, len = 0, cap = 0;
        size_t size() const { return len; }
        size_t capacity() const { return store.use_count() == 1 && off == 0 ? cap : 0; }
        uint8_t* data() { return store->data() + off; }
        const uint8_t* data() const { return store->data() + off; }
        bool borrowed() const { return store.use_count() != 1; }
    };
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& segment : self.segments_) {
            if (!count) break;
            const size_t end = begin + segment.size();
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                if constexpr (!std::is_const<Self>::value)
                    if (segment.borrowed()) return false;   // never write through another owner's bytes
                if (!fn(segment.data() + offset - begin, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Segment> segments_;
    size_t size_ = 0;
};

} // namespace strata::core
