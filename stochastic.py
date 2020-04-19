from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def main():
    # PARAMS
    delta = 0.1
    T = 25.0
    N = 5000
    dt = T/N

    # CIR Params
    # P(t,T) = A(t,T)exp(-B(t,T)r_t)
    a = 0.01  # speed of reversion to b (mean)
    b = 0.001
    sig = 0.0074  # vol

    def A(t, T, a, b, sig):
        h = sqrt(a**2 + 2*sig**2)
        top = 2*h*np.exp(0.5*(a+h)*(T-t)/2)
        bottom = 2*h+(a+h)*(np.exp((T-t)*h)-1)
        return (top/bottom)**(2*a*b/sig**2)

    def B(t, T, a, b, sig):
        h = sqrt(a**2 + 2*sig**2)
        top = 2*np.exp((T-t)*h-1)
        bottom = 2*h+(a+h)*((np.exp(T-t)*h)-1)
        return top/bottom

    def P(t, T, r, a, b, sig):
        P = A(t, T, a, b, sig)*np.exp(-1*B(t, T, a, b, sig)*r)
        return P

    # number of processes
    m = 1

    # array for each proc.
    x = np.empty((m, N+1))

    # IC
    x[:, 0] = 2

    brownian(x[:, 0], N, dt, delta, out=x[:, 1:])

    t = np.linspace(0, N*dt, N+1)

    for k in range(m):
        plt.plot(t, x[k])
        plt.plot(t, P(t, T, x[k]/100, a, b, sig))
    plt.xlabel('t')
    plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    main()
