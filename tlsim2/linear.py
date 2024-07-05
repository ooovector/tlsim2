from abc import abstractmethod
from typing import Iterable
import numpy as np


class CircuitElement:
    @abstractmethod
    def get_element_licri(self):
        pass

    @abstractmethod
    def get_terminal_names(self):
        pass


class LinearElement:
    def split(self, splitting):
        pass


class NonlinearElement(CircuitElement):
    @abstractmethod
    def get_lagrangian_series(self, order):
        """
        Get the coefficients for the series expansion of the current element's phase functions
        :param order:
        :return:
        """
        pass

    def get_element_licri(self):
        li, c, ri = self.get_lagrangian_series(order=2)
        return li, c, ri

    @abstractmethod
    def get_terminal_names(self):
        pass


class SplitElementPart(LinearElement, CircuitElement):
    def __init__(self, name, element: CircuitElement, terminals_a: Iterable, terminals_b: Iterable):
        '''

        :param name:
        :param element:
        :param terminals_a:
        :param terminals_b:
        '''
        self.source = element
        self.terminals_a = terminals_a
        self.terminals_b = terminals_b
        self.name = name

    def get_element_licri(self):
        sli, sc, sri = self.source.get_element_licri()
        terminals = self.get_terminal_names()

        li = np.zeros((len(terminals), len(terminals)), sli.dtype)
        c = np.zeros((len(terminals), len(terminals)), sc.dtype)
        ri = np.zeros((len(terminals), len(terminals)), sri.dtype)

        source_terminal_indeces_a = [self.source.get_terminal_names().index(t) for t in self.terminals_a]
        source_terminal_indeces_b = [self.source.get_terminal_names().index(t) for t in self.terminals_b]
        terminal_indeces_a = [terminals.index(t) for t in self.terminals_a]
        terminal_indeces_b = [terminals.index(t) for t in self.terminals_b]
        li[terminal_indeces_a, terminal_indeces_b] = sli[source_terminal_indeces_a, source_terminal_indeces_b]
        li[terminal_indeces_b, terminal_indeces_a] = sli[source_terminal_indeces_b, source_terminal_indeces_a]
        c[terminal_indeces_a, terminal_indeces_b] = sc[source_terminal_indeces_a, source_terminal_indeces_b]
        c[terminal_indeces_b, terminal_indeces_a] = sc[source_terminal_indeces_b, source_terminal_indeces_a]
        ri[terminal_indeces_a, terminal_indeces_b] = sri[source_terminal_indeces_a, source_terminal_indeces_b]
        ri[terminal_indeces_b, terminal_indeces_a] = sri[source_terminal_indeces_b, source_terminal_indeces_a]

        return li, c, ri

    def get_terminal_names(self):
        terminals = [t for t in self.terminals_a] + [t for t in self.terminals_b if t not in self.terminals_a]
        # TODO: check if all terminals actually are in the source element
        return terminals

