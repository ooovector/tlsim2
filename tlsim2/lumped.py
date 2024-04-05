import numpy as np
from scipy.constants import hbar, h, e


class LumpedTwoTerminal:
    def __init__(self, name, type_: str = None, l: float = np.inf, c: float = 0, r: float = np.inf):
        self.l = l
        self.c = c
        self.r = r
        self.name = name
        self.type_ = type_
        self.terminal_names = ['i', 'o']
        self.stationary_phases = {'i': 0, 'o': 0}

        self.lumped = True

    def get_element_licri(self):
        li = np.asarray([[1 / self.l, -1 / self.l], [-1 / self.l, 1 / self.l]])
        c = np.asarray([[self.c, -self.c], [-self.c, self.c]])
        ri = np.asarray([[1 / self.r, -1 / self.r], [-1 / self.r, 1 / self.r]])
        return li, c, ri

    def get_terminal_names(self):
        return self.terminal_names


class Capacitor(LumpedTwoTerminal):
    def __init__(self, name, c: float = 0):
        LumpedTwoTerminal.__init__(self, name, 'C')
        self.c = c


class Inductor(LumpedTwoTerminal):
    def __init__(self, name, l: float = np.inf):
        LumpedTwoTerminal.__init__(self, name, 'L')
        self.l = l


class Resistor(LumpedTwoTerminal):
    def __init__(self, name, r: float = np.inf):
        LumpedTwoTerminal.__init__(self, name, 'R')
        self.r = r


class JosephsonJunction(LumpedTwoTerminal):
    """
    JosephsonJunction is a nonlinear element with energy E = E_J(1 − cos φ).
    However, in approximation it can be represented as element with linear inductance L_J = Φ_0 / (2 pi I_c),
    where I_c is a critical current.
    """

    def __init__(self, name, ej: float = 0):
        LumpedTwoTerminal.__init__(self, name, 'JJ')
        self.ej = ej
        self.l = self.l_lin()

    def l_lin(self):
        """
        Returns corresponding non-linear inductance
        """
        phi_0 = hbar / (2 * e)
        stationary_phase = self.stationary_phases['i'] - self.stationary_phases['o']
        return phi_0 ** 2 / (self.ej * np.cos(stationary_phase))  # linear part of JJ
