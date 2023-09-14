from __future__ import annotations
from poly_exponent import PolyExponent, poly_exponent_complex
from fractions import Fraction
from basis import LinearHull
from typing import Iterable, Union, Mapping
from field_distribution import FieldDistribution
import numpy as np
from functools import cache

@cache
def default_tl_basis(n: int = 1, n_harmonics: int = 4):
    """
    Create list of
    :param n: number of conductor in multi-conductor TL
    :param n_taylor: number of polynomial coefficients (constant, linear, quadratice etc.) per conductor
    :param n_harmonics: number of harmonic coefficients (quarter-wavelength and higher)
    :return: list of basis functions
    """
    basis_generator = []
    basis = []
    cmplx = poly_exponent_complex

    o = PolyExponent({(0, 0): 0.5, (0, 1):  1})
    i = PolyExponent({(0, 0): 0.5, (0, 1): -1})

    basis_generator.append(i)
    basis_generator.append(o)

    for harm_id in range(1, n_harmonics+1):
        # sine wave start having a zero at -0.5 and a 1 at 0.5 which is
        c1 = cmplx((1 + 1j)/np.sqrt(2)) ** harm_id / 2
        c2 = cmplx((1 - 1j)/np.sqrt(2)) ** harm_id / 2
        harm1 = PolyExponent({(cmplx(0,  harm_id * np.pi / 2), 0):  c1,
                              (cmplx(0, -harm_id * np.pi / 2), 0):  c2})
        harm2 = PolyExponent({(cmplx(0,  harm_id * np.pi / 2), 0): -c1 * 1j,
                              (cmplx(0, -harm_id * np.pi / 2), 0): c2 * 1j})

        harm1 -= (harm1.eval(-0.5) * i + harm1.eval(0.5) * o)
        harm2 -= (harm2.eval(-0.5) * i + harm2.eval(0.5) * o)

        norm1 = np.sqrt(harm1.conjugate() @ harm1 * 3)
        harm1 = harm1 * (1 / norm1)
        norm2 = np.sqrt(harm2.conjugate() @ harm2 * 3)
        harm2 = harm2 * (1 / norm2)

        basis_generator.append(harm1)
        basis_generator.append(harm2)

    for conductor_id in range(n):
        for gen in basis_generator[:2]:
            basis.append(TLFieldDistribution([gen if conductor_id == j else PolyExponent({}) for j in range(n)]))
    for conductor_id in range(n):
        for gen in basis_generator[2:]:
            basis.append(TLFieldDistribution([gen if conductor_id == j else PolyExponent({}) for j in range(n)]))

    return LinearHull(basis)


def default_tl_basis_nonortho(n: int = 1, n_taylor: int = 2, n_harmonics: int = 4):
    """
    Create list of
    :param n: number of conductor in multi-conductor TL
    :param n_taylor: number of polynomial coefficients (constant, linear, quadratice etc.) per conductor
    :param n_harmonics: number of harmonic coefficients (quarter-wavelength and higher)
    :return: list of basis functions
    """
    basis_generator = []
    basis = []
    cmplx = poly_exponent_complex
    for poly_id in range(n_taylor):
        # constant and linear increase
        basis_generator.append(PolyExponent({(0, poly_id): (2 ** (poly_id - 1/2)) * np.sqrt(2 * poly_id + 1)}))

    for harm_id in range(2, n_harmonics+2):
        # sine wave start having a zero at -0.5 and a 1 at 0.5 which is
        c1 = cmplx((1+1j)/np.sqrt(2))**harm_id/2
        c2 = cmplx((1-1j)/np.sqrt(2))**harm_id/2
        basis_generator.append(PolyExponent({(cmplx(0, harm_id*np.pi/2), 0): c1,
                                             (cmplx(0, -harm_id*np.pi/2), 0): c2}))
        basis_generator.append(PolyExponent({(cmplx(0, harm_id*np.pi/2), 0): c1/1j,
                                             (cmplx(0, -harm_id*np.pi/2), 0): -c2/1j}))

    for conductor_id in range(n):
        for gen in basis_generator:
            basis.append(TLFieldDistribution([gen if conductor_id == j else PolyExponent({}) for j in range(n)]))

    return LinearHull(basis)


class TLFieldDistribution(FieldDistribution):
    def __init__(self, elements: Iterable[PolyExponent]):
        self.elements = [c for c in elements]

    def __repr__(self):
        return f'TLFieldDistribution with {len(self.elements)} conductors: '+'\n'.join(
            el.__repr__() for el in self.elements)

    def __add__(self, other: TLFieldDistribution):
        if other != 0:
            return TLFieldDistribution([c1 + c2 for c1, c2 in zip(self.elements, other.elements)])
        else:
            return self

    def __radd__(self, other: TLFieldDistribution):
        return self + other

    def __sub__(self, other: TLFieldDistribution):
        return TLFieldDistribution([c1 - c2 for c1, c2 in zip(self.elements, other.elements)])

    def __mul__(self, other: Union[float, complex, TLFieldDistribution]):
        if hasattr(other, 'elements'):
            return TLFieldDistribution([c1 * c2 for c1, c2 in zip(self.elements, other.elements)])
        else:
            return TLFieldDistribution([other * c for c in self.elements])

    def __rmul__(self, other: Union[float, complex, TLFieldDistribution]):
        if hasattr(other, 'elements'):
            return TLFieldDistribution([c1 * c2 for c1, c2 in zip(self.elements, other.elements)])
        else:
            return TLFieldDistribution([other * c for c in self.elements])

    def __matmul__(self, other: TLFieldDistribution):
        return sum([e1@e2 for e1, e2 in zip(self.elements, other.elements)])

    def conjugate(self) -> TLFieldDistribution:
        return TLFieldDistribution([c.conjugate() for c in self.elements])

    def eval_terminals(self) -> Mapping[str, complex]:
        fields = {}
        for conductor_id, conductor in enumerate(self.elements):
            fields['i' + str(conductor_id)] = conductor.eval(-0.5)
            fields['o' + str(conductor_id)] = conductor.eval(0.5)

        return fields

    def copy(self) -> TLFieldDistribution:
        return TLFieldDistribution([c.copy() for c in self.elements])

    def derivative(self) -> TLFieldDistribution:
        return TLFieldDistribution([c.derivative() for c in self.elements])

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.elements)


class MultiTransmissionLine:
    def __init__(self, name, n, l, ll, cl, rl=None, gl=None, basis=None):
        self.ll = ll
        self.cl = cl
        self.rl = rl
        self.gl = gl
        self.l = l
        self.name = name
        self.n = n
        self.basis = basis
        if basis is None:
            basis = default_tl_basis(n=self.n)
        self.set_basis(basis)

    def set_basis(self, basis):
        self.basis = basis

    def get_element_licri(self):
        # print (self.basis.basis_derivative_products, np.linalg.pinv(self.ll))
        li = np.einsum('ikjl,kl->ij', self.basis.basis_derivative_products, np.linalg.pinv(self.ll)) / self.l
        c = np.einsum('ikjl,kl->ij', self.basis.basis_products, self.cl) * self.l
        #         ri = np.trace(np.linalg.pinv(self.rl)@self.basis_derivative_products)*self.l
        return li, c, np.zeros_like(li)  # ri

    def get_terminal_names(self):
        terminal_names = []
        for i in range(self.n):
            terminal_names.append(f'i{i}')
            terminal_names.append(f'o{i}')
        for i in range(len(self.basis.elements) - 2 * self.n):
            terminal_names.append(f'int{i}')
        return terminal_names

    # def get_terminal_modes(self):
    #     return self.basis_terminals

