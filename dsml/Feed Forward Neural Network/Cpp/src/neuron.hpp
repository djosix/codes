#pragma once

#include "utils.hpp"
#include "activate.hpp"

using namespace std;

class Neuron {
public:
    Neuron();
    Neuron(int, double, Function*);
    ~Neuron();
    double predict(Vector);
    void train(double);
    Vector errors;
private:
    int n_i;
    double bias, sum, output, alpha;
    Vector weights, inputs, adagrad;
    Function *activate;
};
