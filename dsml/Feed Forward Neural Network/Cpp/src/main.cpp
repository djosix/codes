#include <iostream>
#include "network.hpp"
#include "neuron.hpp"
#include "activate.hpp"
#include "error.hpp"

using namespace std;


void test_nn() {
    /* Prime number training set X */
    Vector X[] = {
        {0, 0, 0}, // 0
        {0, 0, 1}, // 1
        {0, 1, 0}, // 2 prime
        {0, 1, 1}, // 3 prime
        {1, 0, 0}, // 4
        {1, 0, 1}, // 5 prime
        {1, 1, 0}, // 6
        {1, 1, 1}  // 7 prime
    };

    /* Prime number training set Y */
    Vector Y[] = {
        {0}, // 0
        {0}, // 1
        {1}, // 2 prime 
        {1}, // 3 prime
        {0}, // 4
        {1}, // 5 prime
        {0}, // 6
        {1}  // 7 prime
    };

    /* Test neural network */
    Sigmoid sigmoid;
    ReLU relu;
    Swish swish;

    double alpha = 0.2;
    double error = 0;

    Network nn( (vector<Layer>) {
        Layer(3, 10, alpha, (Function*) &swish),
        Layer(10, 10, alpha, (Function*) &swish),
        Layer(10, 10, alpha, (Function*) &swish),
        Layer(10, 1, alpha, (Function*) &sigmoid)
    });

    for(int n = 0; n < 500; n++) {
        Vector x = X[n % 8], y = Y[n % 8];
        Vector p = nn.forward(x);
        nn.backward(mean_square_gradient(y, p));
        if (n == 0) error = mean_square_error(y, p);
        else error = (error * n + mean_square_error(y, p)) / (n + 1);
        printf("Error: %.3f\n", error);
    }

    for(int i = 0; i < 8; i++) {
        Vector x = X[i], y = Y[i];
        double p = nn.forward(x)[0];
        if(int(p > 0.5) == y[0]) cout << "[v]  ";
        else cout << "[ ]  ";
        cout << "target: " << y[0] << "  "
             << "predict: " << int(p>0.5) << "  "
             << "output: " << p << endl;
    }
}

void test_neuron() {
    /* XOR training set X */
    Vector X[] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    /* XOR training set Y */
    Vector Y[] = {
        {0},
        {1},
        {1},
        {0}
    };
    
    /* Test a neuron */
    Sigmoid sigmoid;
    Neuron n(2, 0.1, (Function*) &sigmoid);
    for(int i = 0; i < 2000; i++) {
        double p = n.predict(X[i % 4]);
        n.train(p - Y[i % 4][0]);
        cout << "target: " << Y[i % 4][0] << "  "
             << "predict: " << int(p > 0.5) << "  "
             << "error: " << mean_square_error(Y[i % 4], {p})
             << endl;
    }
}



int main() {
    test_nn();
    // test_neuron();
}
