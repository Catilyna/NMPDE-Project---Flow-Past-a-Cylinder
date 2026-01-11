#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer {
public:
    using clock = std::chrono::steady_clock;

    explicit ScopedTimer(std::string name)
        : name_(std::move(name)), start_(clock::now()) {}

    ~ScopedTimer() {
        auto end = clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << " took "
                  << elapsed.count() << " s\n";
    }

private:
    std::string name_;
    clock::time_point start_;
};