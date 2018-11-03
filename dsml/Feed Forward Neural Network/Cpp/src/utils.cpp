#include <cstdlib>
#include <ctime>
#include <cassert>

#include "utils.hpp"

double
uniform (double low, double high)
{
    assert(high > low);
    static bool seeded = false;
    if(!seeded) {
        srand(time(NULL));
        seeded = true;
    }
    double width = high - low;
    return rand() * width / double(RAND_MAX) + low;
}