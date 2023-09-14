import numpy as np


class LumpedTwoTerminal:
    def __init__(self, name, l: float = np.inf, c: float = 0, r: float = np.inf):
        self.l = l
        self.c = c
        self.r = r
        self.name = name
        self.terminal_names = ['i', 'o']

    def get_element_licri(self):
        li = np.asarray([[1 / self.l, -1 / self.l], [-1 / self.l, 1 / self.l]])
        c  = np.asarray([[self.c, -self.c], [-self.c, self.c]])
        ri = np.asarray([[1 / self.r, -1 / self.r], [-1 / self.r, 1 / self.r]])
        return li, c, ri

    def get_terminal_names(self):
        return self.terminal_names



