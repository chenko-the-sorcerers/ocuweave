#pragma once
#include <cstddef>
#include <cstdint>
namespace ocuweave {
struct ConfusionCounts { std::uint64_t tp=0, fp=0, fn=0, tn=0; };
ConfusionCounts confusion_counts(const unsigned char* prediction, const unsigned char* target,
                                const unsigned char* valid, std::size_t count);
}
