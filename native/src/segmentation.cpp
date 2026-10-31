#include "ocuweave/segmentation.hpp"
#include <stdexcept>
namespace ocuweave {
ConfusionCounts confusion_counts(const unsigned char* prediction, const unsigned char* target,
                                const unsigned char* valid, std::size_t count) {
    if (!prediction || !target || !valid || count==0) throw std::invalid_argument("nonempty arrays required");
    ConfusionCounts out;
    for (std::size_t i=0; i<count; ++i) {
        if (valid[i]>1 || prediction[i]>1 || target[i]>1) throw std::invalid_argument("binary arrays required");
        if (!valid[i]) continue;
        if (prediction[i] && target[i]) ++out.tp;
        else if (prediction[i]) ++out.fp;
        else if (target[i]) ++out.fn;
        else ++out.tn;
    }
    if (out.tp+out.fp+out.fn+out.tn==0) throw std::invalid_argument("no valid pixels");
    return out;
}
}
