#include "ocuweave/segmentation.hpp"
#include <stdexcept>
int main() {
    unsigned char p[]={1,1,0,0,1}, y[]={1,0,1,0,0}, v[]={1,1,1,1,0};
    const auto c=ocuweave::confusion_counts(p,y,v,5);
    if(c.tp!=1 || c.fp!=1 || c.fn!=1 || c.tn!=1) return 1;
    p[0]=2;
    try {ocuweave::confusion_counts(p,y,v,5); return 2;} catch(const std::invalid_argument&) {}
    return 0;
}
