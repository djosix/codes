#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "neuron.hpp"

using namespace std;

class Network {
public:
    Network();
    Network(vector<int>, double, Function*);
    Network(vector<Layer>);
    ~Network();
    Vector forward(Vector input);
    Vector backward(Vector dEdo);
private:
    vector<Layer> layers;
};