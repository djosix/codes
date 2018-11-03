#include "error.hpp"


/* mean square error */
double
mean_square_error (
        Vector y_hat, // target y
        Vector y) // predicted y
{
    double error = 0;
    for(int i = 0; i < y_hat.size(); i++) {
        double diff = y_hat[i] - y[i];
        error += diff * diff;
    }
    return error / y_hat.size() / 2;
}

/* mean square gradient */
Vector
mean_square_gradient (
        Vector y_hat, // target y
        Vector y) // predicted y
{
    Vector errors(y_hat.size());
    for(int i = 0; i < y.size(); i++) {
        errors[i] = y[i] - y_hat[i];
    }
    return errors;
}