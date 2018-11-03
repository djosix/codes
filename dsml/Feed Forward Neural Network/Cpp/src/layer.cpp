#include "layer.hpp"
#include "neuron.hpp"

using namespace std;


Layer::Layer () { }
Layer::~Layer () { }

/* neural layer construct */
Layer::Layer (
        int n_i, // num inputs
        int n_o, // num outputs
        double alpha, // learning rate
        Function *activate) // activation function
{
    this->n_i = n_i;
    this->n_o = n_o;
    this->neurons = vector<Neuron>(n_o);
    for(auto &n : this->neurons) {
        n = Neuron(n_i, alpha, activate);
    }
}

/* neural layer feed forward */
Vector
Layer::forward (Vector inputs)
{
    // feed inputs and return outputs
    Vector outputs(this->n_o);
    for(int i = 0; i < this->n_o; i++) {
        outputs[i] = this->neurons[i].predict(inputs);
    }
    return outputs;
}

/* neural layer backpropagate */
Vector
Layer::backward (Vector dEdo)
{
    // propagate output errors, return input errors
    Vector errors(this->n_i, 0);
    for(int i = 0; i < this->n_o; i++) {
        
        // dEdo[i] = partial derivative of total error wrt output[i]
        this->neurons[i].train(dEdo[i]);

        // sum up input errors of all neurons
        for(int j = 0; j < this->n_i; j++) {
            errors[j] += this->neurons[i].errors[j];
        }
    }
    return errors;
}