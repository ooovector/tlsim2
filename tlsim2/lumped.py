import numpy as np
from scipy.constants import hbar, h, e
from .linear import LinearElement


class LumpedTwoTerminal(LinearElement):
    def __init__(self, name, type_: str = None, l: float = np.inf, c: float = 0, r: float = np.inf, coupler_hint=False):
        self.l = l
        self.c = c
        self.r = r
        self.name = name
        self.terminal_names = ['i', 'o']
        self.coupler_hint = coupler_hint

        self.type_ = type_
        self.terminal_names = ['i', 'o']
        self.stationary_phases = {'i': 0, 'o': 0}

        self.lumped = True

    def get_element_licri(self):
        li = np.asarray([[1 / self.l, -1 / self.l], [-1 / self.l, 1 / self.l]])
        c  = np.asarray([[self.c, -self.c], [-self.c, self.c]])
        ri = np.asarray([[1 / self.r, -1 / self.r], [-1 / self.r, 1 / self.r]])
        return li, c, ri

    def get_terminal_names(self):
        return self.terminal_names

    def get_coupling_hints(self):
        if self.coupler_hint:
            return [['i'], ['o']]
        else:
            return [['i', 'o']]

    class Capacitor(LumpedTwoTerminal):
        """
        Lumped two terminal capacitor element with capacitance c in F
        """
        def __init__(self, name, c: float = 0):
            LumpedTwoTerminal.__init__(self, name, 'C')
            self.c = c


    class Inductor(LumpedTwoTerminal):
        """
        Lumped two terminal inductor element with inductance c in H
        """
        def __init__(self, name, l: float = np.inf):
            LumpedTwoTerminal.__init__(self, name, 'L')
            self.l = l


    class Resistor(LumpedTwoTerminal):
        def __init__(self, name, r: float = np.inf):
            LumpedTwoTerminal.__init__(self, name, 'R')
            self.r = r


    class JosephsonJunction(LumpedTwoTerminal):
        """
        JosephsonJunction is a nonlinear element with nonlinear energy E = - E_J cos φ, where E_J in J.
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

        def nonlinear_energy_term(self, node_phases):
            if len(node_phases) != 2:
                raise Exception('ConnectionError',
                                'Josephson junction {0} has {1} nodes connected instead of 2.'.format(self.name,
                                                                                                      len(node_phases)))
            # return self.ej * (1 - np.cos(node_phases[0] - node_phases[1]))
            return self.ej * (1 - np.cos(node_phases[0] - node_phases[1])) - self.ej * (
                    node_phases[0] - node_phases[1]) ** 2 / 2



