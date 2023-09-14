import unittest
from tlsim2.basis import LinearHull
from tlsim2.tl import default_tl_basis
import numpy as np
from matplotlib import pyplot as plt


class TLBasisTest(unittest.TestCase):
    def test_scalar_product(self):
        default_basis = default_tl_basis(n=1)
        print('len(default_basis)', len(default_basis.elements))
        print(default_basis.elements[0]@default_basis.elements[0])

    def test_gram_schmidt(self):
        default_basis = default_tl_basis(n=1)

        old_prods = np.zeros((len(default_basis.elements), len(default_basis.elements)), dtype=complex)
        for gs_el1_id, gs_el1 in enumerate(default_basis.elements):
            for gs_el2_id, gs_el2 in enumerate(default_basis.elements):
                old_prods[gs_el1_id, gs_el2_id] = gs_el1.conjugate()@gs_el2

        gs_basis = default_basis.gram_schmidt()[0]
        new_prods = np.zeros((len(gs_basis.elements), len(gs_basis.elements)), dtype=complex)
        for gs_el1_id, gs_el1 in enumerate(gs_basis.elements):
            for gs_el2_id, gs_el2 in enumerate(gs_basis.elements):
                new_prods[gs_el1_id, gs_el2_id] = gs_el1.conjugate()@gs_el2
        #         if gs_el1_id != gs_el2_id and np.abs(new_prods[gs_el1_id, gs_el2_id]) > 1e-9:
        #             print(gs_el1_id, gs_el2_id, np.abs(new_prods[gs_el1_id, gs_el2_id]))
        #             print(gs_el1)
        #             print(gs_el2)
        #
        #             # plt.plot(gs_el1)
        #
        # print(np.round(old_prods, 7))
        # print(np.round(new_prods, 7))

    def test_orthogonalize(self):
        default_basis = default_tl_basis(n=1)
        print('Testing orthogonalize')
        # orthogonal_basis = default_basis.orthogonalize()
        old_prods = np.zeros((len(default_basis.elements), len(default_basis.elements)), dtype=complex)
        for gs_el1_id, gs_el1 in enumerate(default_basis.elements):
            for gs_el2_id, gs_el2 in enumerate(default_basis.elements):
                old_prods[gs_el1_id, gs_el2_id] = gs_el1.conjugate()@gs_el2

        gs_basis = default_basis.orthogonalize()

        new_prods = np.zeros((len(gs_basis.elements), len(gs_basis.elements)), dtype=complex)
        for gs_el1_id, gs_el1 in enumerate(gs_basis.elements):
            for gs_el2_id, gs_el2 in enumerate(gs_basis.elements):
                new_prods[gs_el1_id, gs_el2_id] = gs_el1.conjugate()@gs_el2
                # if gs_el1_id != gs_el2_id and np.abs(new_prods[gs_el1_id, gs_el2_id]) > 1e-9:
                #     print(gs_el1_id, gs_el2_id, np.abs(new_prods[gs_el1_id, gs_el2_id]))
                #     print(gs_el1)
                #     print(gs_el2)

                    # plt.plot(gs_el1)

        print ('initial basis length: ', len(default_basis.elements))
        print ('orthogonalized basis length: ', len(gs_basis.elements))
        # xlist = np.linspace(-0.5, 0.5, 101)
        # fig, axes = plt.subplots(2, 1)
        # for el in gs_basis.elements:
        #     # print(el)
        #     el_y = []
        #     for x in xlist:
        #         el_y.append(el.elements[0].eval(x))
        #     axes[0].plot(xlist, np.real(el_y))
        #     axes[1].plot(xlist, np.imag(el_y))
        # plt.show()


class BasisTest(unittest.TestCase):
    def test_orthogonalize(self):
        vectors = [np.asarray([1, 1j, 0], dtype=complex),
                   np.asarray([1, 0, 0], dtype=complex),
                   np.asarray([1, 0, 1], dtype=complex),
                   np.asarray([1, -1j, 0], dtype=complex)]
        basis = LinearHull(vectors)
        basis_orthogonal, transform = basis.gram_schmidt()

        product = np.zeros((len(basis_orthogonal.elements), len(basis_orthogonal.elements)), dtype=complex)
        # print('Initial basis: ', vectors)
        # print('Orthogonalized basis: ', basis_orthogonal)
        # print('Transform: ', transform)
        for el1_id, el1 in enumerate(basis_orthogonal.elements):
            for el2_id, el2 in enumerate(basis_orthogonal.elements):
                product[el1_id, el2_id] = el1.conjugate()@el2

        np.allclose(product, np.identity(min(np.asarray(vectors).shape)))

    def test_orthogonalize_terminals(self):
        default_basis = default_tl_basis(n=4, n_harmonics=3)
        # print('Default basis: ', default_basis)
        # transform = default_basis.gram_schmidt_terminals(['o0', 'i0', 'o1', 'i1', 'o2', 'i2', 'o3', 'i3'])
        # print('Gram-Schmidt terminals: ', transform)
        print('Terminal phases: ', '\n'.join([str(el.eval_terminals()) for el in default_basis.elements]))


if __name__ == '__main__':
    unittest.main()
