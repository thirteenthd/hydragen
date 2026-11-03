// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Cross-Platform OS & Dynamic Library Linking Macros
#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>

    #define HYDRAGEN_EXPORT __declspec(dllexport)
    #define HYDRAGEN_IMPORT __declspec(dllimport)

    typedef HMODULE platform_lib_handle_t;

    inline platform_lib_handle_t platform_dlopen(const char* path) {
        return LoadLibraryA(path);
    }

    inline void* platform_dlsym(platform_lib_handle_t handle, const char* symbol) {
        return reinterpret_cast<void*>(GetProcAddress(handle, symbol));
    }

    inline int platform_dlclose(platform_lib_handle_t handle) {
        return FreeLibrary(handle) ? 0 : -1;
    }

    inline std::string platform_dlerror() {
        DWORD err = GetLastError();
        if (err == 0) return "";
        char buf[256];
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            buf, sizeof(buf), NULL
        );
        return std::string(buf);
    }

    #define HYDRAGEN_LIB_EXT ".dll"

#else // Linux, macOS, UNIX
    #include <dlfcn.h>

    #define HYDRAGEN_EXPORT __attribute__((visibility("default")))
    #define HYDRAGEN_IMPORT

    typedef void* platform_lib_handle_t;

    inline platform_lib_handle_t platform_dlopen(const char* path) {
        return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    }

    inline void* platform_dlsym(platform_lib_handle_t handle, const char* symbol) {
        return dlsym(handle, symbol);
    }

    inline int platform_dlclose(platform_lib_handle_t handle) {
        return dlclose(handle);
    }

    inline std::string platform_dlerror() {
        const char* err = dlerror();
        return err ? std::string(err) : "";
    }

    #if defined(__APPLE__)
        #define HYDRAGEN_LIB_EXT ".dylib"
    #else
        #define HYDRAGEN_LIB_EXT ".so"
    #endif

#endif

#ifdef __cplusplus
    #define HYDRAGEN_C_API extern "C" HYDRAGEN_EXPORT
#else
    #define HYDRAGEN_C_API HYDRAGEN_EXPORT
#endif
