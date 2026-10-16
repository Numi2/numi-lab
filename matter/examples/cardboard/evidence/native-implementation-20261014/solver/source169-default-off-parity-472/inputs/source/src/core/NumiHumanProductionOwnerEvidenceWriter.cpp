#include "metalrobo/NumiHumanProductionOwnerEvidenceWriter.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace metalrobo {
namespace {

class Descriptor final {
public:
    Descriptor() = default;
    explicit Descriptor(const int value) noexcept : value_(value) {}
    ~Descriptor() {
        if (value_ >= 0) (void)::close(value_);
    }

    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;

    Descriptor(Descriptor&& other) noexcept : value_(other.release()) {}
    Descriptor& operator=(Descriptor&& other) noexcept {
        if (this == &other) return *this;
        if (value_ >= 0) (void)::close(value_);
        value_ = other.release();
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return value_; }
    [[nodiscard]] bool valid() const noexcept { return value_ >= 0; }

    [[nodiscard]] int release() noexcept {
        const int result = value_;
        value_ = -1;
        return result;
    }

    void reset(const int value = -1) noexcept {
        if (value_ >= 0) (void)::close(value_);
        value_ = value;
    }

private:
    int value_ = -1;
};

[[nodiscard]] bool failErrno(
    std::string& error,
    const char* message,
    const int errorNumber = errno
) {
    error = std::string(message) + ": " + std::strerror(errorNumber);
    return false;
}

[[nodiscard]] bool syncDirectory(
    const int descriptor,
    std::string& error,
    const char* message
) {
    if (::fsync(descriptor) == 0) return true;
    return failErrno(error, message);
}

[[nodiscard]] bool ensureDurableDirectory(
    const std::filesystem::path& directory,
    Descriptor& result,
    std::string& error
) {
    if (directory.empty()) {
        error = "production-owner evidence directory is empty";
        return false;
    }
    constexpr int directoryOpenFlags =
        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
    Descriptor current(::open(directory.is_absolute() ? "/" : ".",
        directoryOpenFlags));
    if (!current.valid()) {
        return failErrno(error,
            "production-owner evidence directory root open failed");
    }

    const auto relative = directory.is_absolute()
        ? directory.relative_path() : directory;
    for (const auto& part : relative) {
        if (part == "..") {
            error = "production-owner evidence directory may not contain '..'";
            return false;
        }
    }
    for (const auto& part : relative) {
        const std::string component = part.string();
        if (component.empty() || component == ".") continue;

        bool missing = false;
        int nextValue = ::openat(
            current.get(), component.c_str(), directoryOpenFlags);
        if (nextValue < 0) {
            const int openError = errno;
            if (openError != ENOENT) {
                return failErrno(error,
                    "production-owner evidence directory component open failed",
                    openError);
            }
            missing = true;
            if (::mkdirat(current.get(), component.c_str(), 0777) != 0 &&
                errno != EEXIST) {
                return failErrno(error,
                    "production-owner evidence directory component create failed");
            }
            nextValue = ::openat(
                current.get(), component.c_str(), directoryOpenFlags);
            if (nextValue < 0) {
                return failErrno(error,
                    "production-owner evidence created directory open failed");
            }
        }
        Descriptor next(nextValue);
        if (missing &&
            (!syncDirectory(next.get(), error,
                 "production-owner evidence created directory sync failed") ||
             !syncDirectory(current.get(), error,
                 "production-owner evidence parent directory sync failed"))) {
            return false;
        }
        current = std::move(next);
    }
    // Keep the descriptor produced by the checked openat traversal. Reopening
    // `directory` by pathname here would let an intermediate component be
    // exchanged for a symlink between validation and publication.
    result = std::move(current);
    error.clear();
    return true;
}

[[nodiscard]] bool syncRegularEvidenceFile(
    const int descriptor,
    std::string& error
) {
    if (::fsync(descriptor) != 0) {
        return failErrno(error,
            "production-owner evidence file sync failed");
    }
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    if (::fcntl(descriptor, F_FULLFSYNC) != 0) {
        return failErrno(error,
            "production-owner evidence file full sync failed");
    }
    return true;
#else
    error = "production-owner evidence requires Darwin F_FULLFSYNC; "
        "plain fsync is not promoted as power-loss durable";
    return false;
#endif
}

} // namespace

bool publishNumiHumanProductionOwnerEvidenceNoReplace(
    const std::filesystem::path& target,
    const std::string_view contents,
    std::string& error
) noexcept {
    try {
        const std::filesystem::path directory = target.parent_path().empty()
            ? std::filesystem::path{"."} : target.parent_path();
        const std::string targetName = target.filename().string();
        if (targetName.empty() || targetName == "." || targetName == "..") {
            error = "production-owner evidence target has no filename";
            return false;
        }
        Descriptor directoryDescriptor;
        if (!ensureDurableDirectory(
                directory, directoryDescriptor, error)) return false;

        static std::atomic<std::uint64_t> nextTemporaryIdentity{1u};
        Descriptor descriptor;
        std::string temporaryName;
        for (std::uint32_t attempt = 0u; attempt < 16u; ++attempt) {
            const std::uint64_t identity =
                nextTemporaryIdentity.fetch_add(1u,
                    std::memory_order_relaxed);
            temporaryName = ".numi-owner-snapshot." +
                std::to_string(static_cast<unsigned long long>(::getpid())) +
                "." + std::to_string(identity) + ".tmp";
            const int value = ::openat(directoryDescriptor.get(),
                temporaryName.c_str(),
                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                0600);
            if (value >= 0) {
                descriptor.reset(value);
                break;
            }
            if (errno != EEXIST) {
                return failErrno(error,
                    "production-owner evidence temporary create failed");
            }
        }
        if (!descriptor.valid()) {
            error = "production-owner evidence temporary namespace exhausted";
            return false;
        }
        const auto discardTemporary = [&]() noexcept {
            descriptor.reset();
            (void)::unlinkat(
                directoryDescriptor.get(), temporaryName.c_str(), 0);
        };

        std::size_t offset = 0u;
        while (offset < contents.size()) {
            const std::size_t remaining = contents.size() - offset;
            const std::size_t chunk = std::min<std::size_t>(remaining,
                static_cast<std::size_t>(
                    std::numeric_limits<ssize_t>::max()));
            const ssize_t written = ::write(
                descriptor.get(), contents.data() + offset, chunk);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                (void)failErrno(error,
                    "production-owner evidence write failed");
                discardTemporary();
                return false;
            }
            offset += static_cast<std::size_t>(written);
        }
        if (!syncRegularEvidenceFile(descriptor.get(), error)) {
            discardTemporary();
            return false;
        }
        const int rawDescriptor = descriptor.release();
        if (::close(rawDescriptor) != 0) {
            const int closeError = errno;
            (void)::unlinkat(
                directoryDescriptor.get(), temporaryName.c_str(), 0);
            return failErrno(error,
                "production-owner evidence file close failed", closeError);
        }

        if (::linkat(directoryDescriptor.get(), temporaryName.c_str(),
                directoryDescriptor.get(), targetName.c_str(), 0) != 0) {
            const int linkError = errno;
            (void)::unlinkat(
                directoryDescriptor.get(), temporaryName.c_str(), 0);
            if (linkError == EEXIST) {
                error = "production-owner evidence target already exists";
                return false;
            }
            return failErrno(error,
                "production-owner evidence no-replace publication failed",
                linkError);
        }
        if (::unlinkat(directoryDescriptor.get(), temporaryName.c_str(), 0) !=
            0) {
            const int unlinkError = errno;
            (void)::fsync(directoryDescriptor.get());
            return failErrno(error,
                "production-owner evidence temporary unlink failed",
                unlinkError);
        }
        if (!syncDirectory(directoryDescriptor.get(), error,
                "production-owner evidence directory sync failed")) {
            return false;
        }
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = std::string(
            "production-owner evidence publication exception: ") +
            exception.what();
        return false;
    } catch (...) {
        error = "production-owner evidence publication unknown exception";
        return false;
    }
}

} // namespace metalrobo
