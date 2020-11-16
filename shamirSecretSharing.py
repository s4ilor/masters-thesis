from __future__ import division
from __future__ import print_function
import random
import functools


# 8th Mersenne Prime is 2**32 -1
# 13th Mersenne Prime is 2**521 - 1

_PRIME = 2 ** 31 - 1
_RINT = functools.partial(random.SystemRandom().randint, 0)


def _eval_at(poly, x, prime):
    """Evaluates polynomial (coefficient tuple) at x, used to generate a
    shamir pool in make_random_shares below.
    """
    accum = 0
    for coeff in reversed(poly):
        accum *= x
        accum += coeff
        accum %= prime
    return accum


def make_random_shares(minimum, shares, prime=_PRIME):
    """
    Generates a random shamir pool, returns the secret and the share
    points.
    """
    if minimum > shares:
        raise ValueError("Pool secret would be irrecoverable.")
    poly = [_RINT(prime - 1) for i in range(minimum)]
    points = [(i, _eval_at(poly, i, prime))
              for i in range(1, shares + 1)]
    return poly[0], points


def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1) this can
    be computed via extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    return last_x, last_y


def _divmod(num, den, p):
    """Compute num / den modulo prime p

    To explain what this means, the return value will be such that
    the following is true: den * _divmod(num, den, p) % p == num
    """
    inv, _ = _extended_gcd(den, p)
    return num * inv


def _lagrange_interpolate(x, x_s, y_s, p):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct"

    def PI(vals):  # upper-case PI -- product of inputs
        accum = 1
        for v in vals:
            accum *= v
        return accum

    nums = []  # avoid inexact division
    dens = []
    for i in range(k):
        others = list(x_s)
        cur = others.pop(i)
        nums.append(PI(x - o for o in others))
        dens.append(PI(cur - o for o in others))
    den = PI(dens)
    num = sum([_divmod(nums[i] * den * y_s[i] % p, dens[i], p)
               for i in range(k)])
    return (_divmod(num, den, p) + p) % p


def recover_secret(shares, prime=_PRIME):
    """
    Recover the secret from share points
    (x, y points on the polynomial).
    """
    if len(shares) < 2:
        raise ValueError("need at least two shares")
    x_s, y_s = zip(*shares)
    return _lagrange_interpolate(0, x_s, y_s, prime)


def main():
    numOfShares = 5
    threshold = 3

    secret, shares = make_random_shares(minimum=threshold, shares=numOfShares)

    print('Secret: ', secret)
    print('\nShares:')
    if shares:
        for share in shares:
            print('  ', share)

    firstShareGroup = shares[:threshold]
    secondShareGroup = shares[-threshold:]
    print('\nSecret recovered from minimum subset of shares:             ',
          recover_secret(firstShareGroup))
    print('Secret recovered from a different minimum subset of shares: ',
          recover_secret(secondShareGroup))

    print('\nAre recovered secrets equal?: '
          + str(recover_secret(firstShareGroup) == recover_secret(secondShareGroup)))


if __name__ == '__main__':
    main()
