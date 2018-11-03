#include <vector>

#include "network.hpp"
#include "layer.hpp"
#include "activate.hpp"

using namespace std;


Network::Network() { }
Network::~Network() { }

/* neural network construct */
Network::Network (
        vector<int> size_layers, // sizes of each layer
        double alpha, // learning rate
        Function *activate) // activation function
{
    int n_layers = size_layers.size() - 1; // input layer has no neuron
    this->layers = vector<Layer>(n_layers);
    for(int i = 0; i < n_layers; i++) {
        int n_i = size_layers[i]; // num inputs
        int n_o = size_layers[i+1]; // num outputs
        this->layers[i] = Layer(n_i, n_o, alpha, activate);
    }
}

/* neural network construction with layers */
Network::Network (vector<Layer> layers)
{
    this->layers = layers;
}

/* neural network feed forward */
Vector
Network::forward (Vector inputs)
{
    // forward data from input layer to output layer
    Vector data = inputs;
    for(auto &l : this->layers) {
        data = l.forward(data);
    }
    return data;
}

/* neural network backpropagate */
Vector
Network::backward (Vector dEdo)
{
    // backpropagate errors from output layer to input layer
    Vector data = dEdo;
    for(auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
        data = (*it).backward(data);
    }
    return data;
}
