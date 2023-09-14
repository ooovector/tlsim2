import unittest
from tlsim2.poly_exponent import PolyExponent
from tlsim2.tl import default_tl_basis
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt


class TLTests(unittest.TestCase):
    def test_default_tl_basis(self):
        default_basis = default_tl_basis(n=1, n_harmonics=1)
        print('Printing default TL basis: ')
        xlist = np.linspace(-0.5, 0.5, 101)
        fig, axes = plt.subplots(4, 1)
        for el in default_basis.elements:
            # print(el)
            el_y = []
            d_y = []
            for x in xlist:
                el_y.append(el.elements[0].eval(x))
                d_y.append(el.elements[0].derivative().eval(x))
            axes[0].plot(xlist, np.real(el_y))
            axes[1].plot(xlist, np.imag(el_y))
            axes[2].plot(xlist, np.real(d_y))
            axes[3].plot(xlist, np.imag(d_y))
        plt.show()
        print('Basis products: ', default_basis.basis_products)
        print('Basis derivative products: ', default_basis.basis_derivative_products)


if __name__ == '__main__':
    unittest.main()
