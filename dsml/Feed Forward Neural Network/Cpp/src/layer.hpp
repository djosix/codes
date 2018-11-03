#pragma once

#include "utils.hpp"
#include "neuron.hpp"

using namespace std;

class Layer {
public:
    Layer();
    Layer(int, int, double, Function*);
    ~Layer();
    Vector forward(Vector);
    Vector backward(Vector);
private:
    int n_i, n_o;
    vector<Neuron> neurons;
};