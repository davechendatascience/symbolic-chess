// math-engine-cpp — entry point
// Delegates everything to the UCI loop in uci.cpp.

#include "uci.h"

int main() {
    math_engine::uci_loop();
    return 0;
}
