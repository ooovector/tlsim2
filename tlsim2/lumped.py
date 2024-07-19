import numpy as np
from .linear import LinearElement


class LumpedTwoTerminal(LinearElement):
    def __init__(self, name, l: float = np.inf, c: float = 0, r: float = np.inf, coupler_hint=False):
        self.l = l
        self.c = c
        self.r = r
        self.name = name
        self.terminal_names = ['i', 'o']
        self.coupler_hint = coupler_hint

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
