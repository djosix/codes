#pragma once

#include "utils.hpp"

class Function {
public:
    Function();
    ~Function();
    virtual double f(double) = 0; // itself
    virtual double d(double) = 0; // derivative
};


class Linear : public Function {
public:
    Linear();
    ~Linear();
    double f(double);
    double d(double);
};

class Sigmoid : public Function {
public:
    Sigmoid();
    ~Sigmoid();
    double f(double);
    double d(double);
};

class Swish : public Function {
public:
    Swish();
    ~Swish();
    double f(double);
    double d(double);
};

class ReLU : public Function {
public:
    ReLU();
    ~ReLU();
    double f(double);
    double d(double);
};
