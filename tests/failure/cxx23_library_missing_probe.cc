// Probe TU for the missing-C++23-facility fixture: compiled at C++17 so the
// sentinel's named-facility static asserts fire instead of the library's own
// opaque #error.
#include "cxx23_features.h"

int main() { return 0; }
