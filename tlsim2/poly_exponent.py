from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Mapping, Tuple, Union
from math import perm


class poly_exponent_complex(complex):
    def eval(self):
        return self


class PolyExponent:
    poly_exp_dict: defaultdict[Tuple[Union[poly_exponent_complex, int], int], Union[int, float, poly_exponent_complex]]

    def __init__(self, poly_exp_dict: Mapping):
        """
        Class that represents a sum of terms of the form
        a_i * exp(k_i*x) * x^(l_i),
        where a_i are the coefficients,
              k_i are the exponent_factors,
              l_i are the poly_exponents
        Args:
            poly_exp_dict: dict[(k_i, l_i): a_i]
        """
        self.poly_exp_dict = defaultdict(int)
        for exp, coef in poly_exp_dict.items():
            k, l = exp
            if l < 0:
                raise ValueError('Only non-negative values of l are allowed')
            self.poly_exp_dict[(poly_exponent_complex(k), int(l))] = coef

    def __add__(self, other) -> PolyExponent:
        result = defaultdict(complex)
        result.update(self.poly_exp_dict)
        if not hasattr(other, 'poly_exp_dict') and other == 0:
            pass
        else:
            for exp, coef in other.poly_exp_dict.items():
                result[exp] += coef
        return PolyExponent(result)

    def __radd__(self, other) -> PolyExponent:
        return self + other

    def __sub__(self, other) -> PolyExponent:
        result = defaultdict(complex)
        result.update(self.poly_exp_dict)
        for exp, coef in other.poly_exp_dict.items():
            result[exp] -= coef
        return PolyExponent(result)

    def __mul__(self, other) -> PolyExponent:
        result = defaultdict(complex)
        if not hasattr(other, 'poly_exp_dict'):
            for exp1, coef1 in self.poly_exp_dict.items():
                result[(exp1[0], exp1[1])] += coef1 * other
        else:
            for exp1, coef1 in self.poly_exp_dict.items():
                for exp2, coef2 in other.poly_exp_dict.items():
                    result[(exp1[0] + exp2[0], exp1[1] + exp2[1])] += coef1 * coef2
        return PolyExponent(result)

    def __rmul__(self, other) -> PolyExponent:
        return self * other

    def derivative(self) -> PolyExponent:
        '''
        Derivate of a polyexponent
        :return:
        '''
        result = defaultdict(poly_exponent_complex)
        for exp, coef in self.poly_exp_dict.items():
            k, l = exp
            if l > 0:
                result[(k, l - 1)] += l * coef
            if k != 0:
                result[(k, l)] += k * coef
        return PolyExponent(result)

    def antiderivate(self) -> PolyExponent:
        '''
        Incomplete intergal of a polyexponent
        :return: PolyExponent antiderivate of self
        '''
        result = defaultdict(poly_exponent_complex)
        for exp, coef in self.poly_exp_dict.items():
            k, l = exp
            if k == 0:
                coef_new = coef / (l + 1)
                l_new = l + 1
                result[(0, l_new)] += coef_new
            else:
                for j in range(0, l+1):  # loop j from 0 to l_i - 1
                    sign = 1 if not j % 2 else -1
                    l_new = l - j
                    k_new = k
                    coef_new = poly_exponent_complex(sign * coef * perm(l, (l - j))) / k ** (j + 1)
                    result[(k_new, l_new)] += coef_new
        return PolyExponent(result)

    def eval(self, x) -> complex:
        """
        Evaluate the PolyExponent at a fixed point
        :param x: Point at which we should evaluate at
        :return: Value
        """

        result = complex()
        for exp, coef in self.poly_exp_dict.items():
            k, l = exp
            k = poly_exponent_complex(k).eval()
            coef = poly_exponent_complex(coef).eval()
            result += coef * np.exp(k * x) * (x ** l)
        return result

    def __repr__(self):
        term_strs = []
        for exp, coef in self.poly_exp_dict.items():
            k, l = exp
            if coef == 0:
                continue

            if k == 0:
                exp_part = f''
            else:
                exp_part = f'* exp({k} * x)'

            if l == 0:
                poly_part = f''
            elif l == 1:
                poly_part = f'* x'
            else:
                poly_part = f'* (x ** {l})'

            term_strs.append(f'{coef} {exp_part} {poly_part}')
        return ' + '.join(term_strs)

    def conjugate(self) -> PolyExponent:
        result = defaultdict(complex)
        for exp, coef in self.poly_exp_dict.items():
            k, l = exp
            coef_conj = coef.conjugate() if hasattr(coef, 'conjugate') else coef
            result[(poly_exponent_complex(k).conjugate(), l)] = coef_conj
        return PolyExponent(result)

    def integrate(self, a, b) -> complex:
        """
        Integrate PolyExponent from a to b analytically, using antiderivative.
        :param a: lower bound for integration
        :param b: upper bound of integration
        :return: Integral from a to b (real evaluated number)
        """
        antiderivative = self.antiderivate()
        return antiderivative.eval(b) - antiderivative.eval(a)

    def __matmul__(self, other) -> complex:
        """
        Compute dot product of two PolyExponents by conjugating the LHS element, multiplying with RHS and integrating
        over [-0.5, 0.5].
        """
        return (self * other).integrate(-0.5, 0.5)

    def copy(self) -> PolyExponent:
        result = defaultdict(complex)
        result.update(self.poly_exp_dict)
        return PolyExponent(result)

