import unittest
from tlsim2.poly_exponent import PolyExponent
import numpy as np


class PolyExponentTest(unittest.TestCase):
    def test_instantization(self):
        PolyExponent({(1, 0): 0})  # add assertion here

    def test_repr(self):
        val1 = PolyExponent({(0.5+1j, 0): 1})  # add assertion here
        val2 = PolyExponent({(0.5+1j, 5): 0.5})  # add assertion here
        print('Testing repr: ')
        print('val1:', val1)
        print('val2:', val2)

    def test_mul(self):
        k1 = 1 + 2j
        l1 = 1
        k2 = 2 + 3j
        l2 = 2
        a = PolyExponent({(k1, l1): 2, (k2, l2): 2})
        print('Testing multiplication: ')
        print('a:', a)
        print('a*a:', a*a)

    def test_conjugate(self):
        k = 2 + 3j
        l = 2
        a = PolyExponent({(k, l): 1j})
        print('Testing conjugate: ')
        print('a:', a)
        print('a.conjugate():', a.conjugate())

    def test_derivative(self):
        k = 1 + 2j
        l = 1
        a = PolyExponent({(k, l): 2+1j})
        a1 = a.derivative()
        print('Testing derivative: ')
        print(f'a: ', a)
        print(f'Derivative of a: ', a1)

    def test_antiderivative(self):
        k = 1 + 2j
        l = 1
        a = PolyExponent({(k, l): 2+1j})
        a1 = a.antiderivate()
        print('Testing antiderivative: ')
        print(f'a: ', a)
        print(f'Antiderivative of a: ', a1)

    def test_antiderivative2(self):
        # import matplotlib
        # from matplotlib import pyplot as plt
        poly_exponent_complex =  PolyExponent({(0, 0): 1, (0, 1): 1j, (1, 0): 1j, (1, 0): -1j})
        # print('Testing integrals: ', poly_exponent_complex)
        sqr = poly_exponent_complex.conjugate()*poly_exponent_complex
        sqr_ad = sqr.antiderivate()
        # print('Self-square: ', sqr)
        # print('Self-square ad: ', sqr_ad)
        # xvals = np.linspace(-0.5, 0.5, 100)
        # y = []
        # y_ad = []
        # for x in np.linspace(-0.5, 0.5, 100):
        #     y.append(sqr.eval(x))
        #     y_ad.append(sqr_ad.eval(x))
        # plt.plot(xvals, np.real(y), label='sqr real')
        # plt.plot(xvals, np.imag(y), label='sqr imag')
        # plt.plot(xvals, np.real(y_ad), label='sqr ad real')
        # plt.plot(xvals, np.imag(y_ad), label='sqr ad imag')
        #
        # plt.legend()
        # plt.show()

    def test_integrals(self):
        basis_size = 2
        poly_basis_elements = [
            PolyExponent({(0, l): 1}) for l in range(basis_size)
        ]
        exp_basis_elements = [
            PolyExponent({(2*np.pi*n*1j, 0): 1}) for n in range(basis_size)
        ]
        scalar_products_exp = np.zeros((basis_size, basis_size), complex)
        scalar_products_poly = np.zeros((basis_size, basis_size), complex)

        for i, el1 in enumerate(exp_basis_elements):
            for j, el2 in enumerate(exp_basis_elements):
                print('Evaluating dot product of harmonics: ', el1, 'and', el2)
                scalar_products_exp[i, j] = (el1.conjugate() * el2).integrate(0, 1)

        for i, el1 in enumerate(poly_basis_elements):
            for j, el2 in enumerate(poly_basis_elements):
                print('Evaluating dot product of powers: ', el1, 'and', el2)
                scalar_products_poly[i, j] = (el1.conjugate() * el2).integrate(0, 1)

        print (scalar_products_exp, scalar_products_poly)

    def test_numpy_with_PolyExponent_dtype(self):
        basis_size = 2
        poly_basis_elements = np.asarray([
            PolyExponent({(0, l): 1}) for l in range(basis_size)
        ])
        exp_basis_elements = np.asarray([
            PolyExponent({(2*np.pi*n*1j, 0): 1}) for n in range(basis_size)
        ])
        print('Testing numy arrays with PolyExponent dtype (element-wise multiplication):')
        print (poly_basis_elements*exp_basis_elements)


if __name__ == '__main__':
    unittest.main()

