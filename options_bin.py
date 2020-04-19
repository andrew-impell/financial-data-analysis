'''
#####
Find optimal pricing for an american option
#####

function americanPut(T, S, K, r, sigma, q, n)
{
  '  T... expiration time
  '  S... stock price
  '  K... strike price
  '  q... dividend yield
  '  n... height of the binomial tree
  deltaT := T / n;
  up := exp(sigma * sqrt(deltaT));
  p0 := (up*exp(-q * deltaT) - exp(-r * deltaT)) / (up^2 - 1);
  p1 := exp(-r * deltaT) - p0;
  ' initial values at time T
  for i := 0 to n {
      p[i] := K - S * up^(2*i - n);
      if p[i] < 0 then p[i] := 0;
  }
  ' move to earlier times
  for j := n-1 down to 0 {
      for i := 0 to j {
          ' binomial value
          p[i] := p0 * p[i+1] + p1 * p[i];
          ' exercise value
          exercise := K - S * up^(2*i - j);
          if p[i] < exercise then p[i] := exercise;
      }
  }
  return americanPut := p[0];
}
'''
import numpy as np
import matplotlib.pyplot as plt

T = 30
S = 100
K = 105
q = 0.01
n = 30
sigma = 1
r = 0.03


def americanPut(T, S, K, r, sigma, q, n):
    p = np.zeros(n)
    deltaT = T/n
    up = np.exp(sigma * np.sqrt(deltaT))
    p0 = (up * np.exp(-q * deltaT) - np.exp(-r * deltaT)) / (up**2 - 1)
    p1 = np.exp(-r * deltaT) - p0
    for i in range(n):
        p[i] = K - S * up**(2*i - n)
        if p[i] < 0:
            p[i] = 0
    for j in range(n-1, 0, -1):
        for i in range(0, j):
            p[i] = p0 * p[i+1] + p1 * p[i]
            exercise = K - S * up**(2*i - j)
            if p[i] < exercise:
                p[i] = exercise
    return p[0]


tree_heights = []

for i in np.arange(0.01, 0.2, 0.001):
    tree_heights.append(americanPut(T, S, K, r, i, q, n))

plt.plot(np.arange(0.01, 0.2, 0.001), tree_heights)
plt.show()
