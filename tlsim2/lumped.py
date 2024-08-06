import numpy as np
from .linear import LinearElement, NonlinearElement
from scipy.constants import hbar, e
from math import factorial


class LumpedTwoTerminal(LinearElement):
    def __init__(self, name, l: float = np.inf, c: float = 0, r: float = np.inf,
                 coupler_hint=False, keep_top_level=False):
        """

        :param name:
        :param l: parallel inductance
        :param c: parallel capacitance
        :param r: parallel resistance
        :param coupler_hint: if set to true, this element will be split across in autosplit
        :param keep_top_level: Flag that will be acknowledged by autosplit, leaving
        the element in the top-level system.
        """

        self.l = l
        self.c = c
        self.r = r
        self.name = name
        self.terminal_names = ['i', 'o']
        self.coupler_hint = coupler_hint
        self.keep_top_level = keep_top_level

    def get_element_licri(self):
        li = np.asarray([[1 / self.l, -1 / self.l], [-1 / self.l, 1 / self.l]])
        c = np.asarray([[self.c, -self.c], [-self.c, self.c]])
        ri = np.asarray([[1 / self.r, -1 / self.r], [-1 / self.r, 1 / self.r]])
        return li, c, ri

    def get_terminal_names(self):
        return self.terminal_names

    def get_coupling_hints(self):
        if self.coupler_hint:
            return [['i'], ['o']]
        else:
            return [['i', 'o']]


class JosephsonJunction(NonlinearElement):
    def __init__(self, name, ej: float, ):
        self.name = name
        self.ej = ej
        self.terminal_names = ['i', 'o']

        self.phi0 = hbar / (2 * e)

    def get_lagrangian_series(self, order: int = 2):
        if order < 2:
            raise ValueError("Order of lagrangian series have to be equal or larger than 2")
        dim = (2, 2, order - 1)
        li, c, ri = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        for i in range(2, order + 1):
            if i % 2 != 0:
                l = np.inf
            else:
                l = self.phi0 ** 2 / self.ej * factorial(i) / i
            li[..., i - 2] = np.asarray([[1 / l, - 1 / l],
                                         [-1 / l, 1 / l]])
        return li, c, ri

    def get_potential_energy(self, phases):
        """For Josephson Junction element potential energy is U(Phi) = EJ (1 - cos(Phi / phi0))"""
        return self.ej * (1 - np.cos((phases[0] - phases[1]) / self.phi0))

    def get_terminal_names(self):
        return self.terminal_names
