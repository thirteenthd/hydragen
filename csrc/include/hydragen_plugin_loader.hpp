// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hydragen_plugin.h"
#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace hydragen {

class HydragenPlugin {
public:
    HydragenPlugin(platform_lib_handle_t handle, const hydragen_plugin_descriptor_t* desc, std::string path)
        : handle_(handle), desc_(desc), path_(std::move(path)) {}

    ~HydragenPlugin() {
        if (handle_) {
            platform_dlclose(handle_);
            handle_ = nullptr;
        }
    }

    // Disable copy
    HydragenPlugin(const HydragenPlugin&) = delete;
    HydragenPlugin& operator=(const HydragenPlugin&) = delete;

    // Enable move
    HydragenPlugin(HydragenPlugin&& other) noexcept
        : handle_(other.handle_), desc_(other.desc_), path_(std::move(other.path_)) {
        other.handle_ = nullptr;
        other.desc_ = nullptr;
    }

    HydragenPlugin& operator=(HydragenPlugin&& other) noexcept {
        if (this != &other) {
            if (handle_) platform_dlclose(handle_);
            handle_ = other.handle_;
            desc_ = other.desc_;
            path_ = std::move(other.path_);
            other.handle_ = nullptr;
            other.desc_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] const hydragen_plugin_descriptor_t* descriptor() const noexcept { return desc_; }
    [[nodiscard]] const hydragen_attention_ops_t* ops() const noexcept { return desc_ ? desc_->attention_ops : nullptr; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }
    [[nodiscard]] const char* name() const noexcept { return desc_ ? desc_->plugin_name : "unknown"; }
    [[nodiscard]] const char* target_arch() const noexcept { return desc_ ? desc_->target_architecture : "generic"; }

private:
    platform_lib_handle_t handle_{nullptr};
    const hydragen_plugin_descriptor_t* desc_{nullptr};
    std::string path_;
};

class HydragenPluginLoader {
public:
    using GetDescriptorFn = const hydragen_plugin_descriptor_t* (*)();

    static std::unique_ptr<HydragenPlugin> load_plugin(const std::string& filepath) {
        platform_lib_handle_t handle = platform_dlopen(filepath.c_str());
        if (!handle) {
            return nullptr;
        }

        auto* get_desc = reinterpret_cast<GetDescriptorFn>(
            platform_dlsym(handle, "hydragen_get_descriptor")
        );

        if (!get_desc) {
            // Not a conforming hydragen plugin or missing descriptor symbol
            platform_dlclose(handle);
            return nullptr;
        }

        const hydragen_plugin_descriptor_t* desc = get_desc();
        if (!desc || desc->abi_version != HYDRAGEN_PLUGIN_ABI_VERSION) {
            // Incompatible ABI version
            platform_dlclose(handle);
            return nullptr;
        }

        return std::make_unique<HydragenPlugin>(handle, desc, filepath);
    }
};

} // namespace hydragen
