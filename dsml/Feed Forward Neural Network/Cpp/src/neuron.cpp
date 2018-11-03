#include <cmath>

#include "utils.hpp"
#include "neuron.hpp"
#include "activate.hpp"

using namespace std;

Neuron::Neuron () { }
Neuron::~Neuron () { }

/* neuron construct */
Neuron::Neuron (
        int n_i, // num inputs
        double alpha, // learning rate
        Function *activate) // activation function
{
    this->n_i = n_i;
    this->weights = Vector(n_i + 1, 0);
    this->adagrad = Vector(n_i + 1, 0);
    this->errors = Vector(n_i, 0);
    this->alpha = alpha;
    this->bias = 1;
    this->activate = activate;

    // initialize weights between -1 and 1 randomly
    for(auto &weight : this->weights) {
        weight = uniform(-.1, .11);
    }
}

/* neuron predict */
double
Neuron::predict (Vector inputs)
{
    this->inputs = inputs;

    // add bias
    this->inputs.push_back(this->bias);

    // sum = weights dot inputs
    this->sum = 0;
    for(int i = 0; i < this->n_i + 1; i++) {
        this->sum += this->weights[i] * this->inputs[i];
    }

    // output = activation(sum)
    this->output = this->activate->f(this->sum);
    return this->output;
}

/* neuron train */
void
Neuron::train (double dEdo)
{
    // do/ds = derivative of activation function
    double dods = this->activate->d(this->sum);

    // chain rule, dE/ds = dE/do * do/ds
    double dEds = dEdo * dods;

    // ps/pw = input
    Vector pspw = this->inputs;

    for(int i = 0; i < this->n_i + 1; i++) {
        // ignore bias
        if(i < this->n_i) {
            // dE/di = dE/ds * w
            this->errors[i] = this->weights[i] * dEds;
        }

        // pE/pw = dE/do * do/ds * ps/pw
        double pEpw = dEds * pspw[i];

        // memorize adagrad
        this->adagrad[i] += pEpw * pEpw;
        double divisor = sqrt(this->adagrad[i]) + 1e-10;

        // gradient descent
        this->weights[i] -= this->alpha * pEpw / divisor;
    }
}
