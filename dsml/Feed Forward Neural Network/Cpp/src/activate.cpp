#include <cmath>
#include "activate.hpp"


Function::Function() {}
Sigmoid::Sigmoid() {}
Linear::Linear() {}
Swish::Swish() {}
ReLU::ReLU() {}

Function::~Function() {}
Sigmoid::~Sigmoid() {}
Linear::~Linear() {}
Swish::~Swish() {}
ReLU::~ReLU() {}



// in file

static inline double
sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


static inline double
d_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}


// public

double
Sigmoid::f(double x)
{
    return sigmoid(x);
}

double
Sigmoid::d(double x)
{
    return d_sigmoid(x);
}

double
ReLU::f (double x)
{
    if(x >= 0) return x;
    else return 0;
}

double
ReLU::d (double x)
{
    if(x >= 0) return 1;
    else return 0;
}

double
Linear::f (double x)
{
    return x;
}

double
Linear::d (double x)
{
    return 1;
}

double
Swish::f (double x)
{
    return x * sigmoid(x);
}

double
Swish::d (double x)
{
    return sigmoid(x) + x * d_sigmoid(x);
}
