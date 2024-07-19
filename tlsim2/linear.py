from abc import abstractmethod, ABC
from typing import Iterable, Mapping, Any
import numpy as np


class CircuitElement:
    @abstractmethod
    def get_element_licri(self):
        pass

    @abstractmethod
    def get_terminal_names(self):
        pass


class LinearElement(CircuitElement, ABC):
    def assign_loose_nodes(self,
                           subsystems: Mapping[Any, Iterable[Any]],
                           coupling_threshold: float = 1e-8) -> Mapping[Any, Iterable[Any]]:
        '''
        Assignes nodes to subsystems using an affinity determination algorithm.
        :param subsystems:
        :param coupling_threshold:
        :return:
        '''
        filled_nodes = set()
        for element_name, nodes in subsystems.items():
            intersection = filled_nodes & set(nodes)
            if len(intersection) != 0:
                raise ValueError('nodes ' + str(intersection) + ' found in several parts of splitting')
            filled_nodes |= set(nodes)
        loose_nodes = set(self.get_terminal_names()) - filled_nodes
        splitting_full = {k: [i for i in v] for k, v in subsystems.items()}

        terminals = self.get_terminal_names()

        li, c, ri = self.get_element_licri()
        li_sqrt_diag = 1 / np.sqrt(np.diag(li))
        li_rel = np.einsum('ij,i,j->ij', li, li_sqrt_diag, li_sqrt_diag)
        c_sqrt_diag = 1 / np.sqrt(np.diag(c))
        c_rel = np.einsum('ij,i,j->ij', c, c_sqrt_diag, c_sqrt_diag)
        ri_sqrt_diag = 1 / np.sqrt(np.diag(ri))
        ri_rel = np.einsum('ij,i,j->ij', ri, ri_sqrt_diag, ri_sqrt_diag)
        while len(loose_nodes) > 0:
            iteration_stalled = True
            splitting_full_next = {k: [i for i in v] for k, v in splitting_full.items()}
            for node in list(loose_nodes):
                node_id = terminals.index(node)
                # compute capacitive and inductive coupling between nodes TODO: only for li and c matrices for now
                nodes_ids = {element_name: [terminals.index(node) for node in nodes]
                             for element_name, nodes in splitting_full.items()}
                couplings = {element_name: np.sum(np.abs(li_rel[node_id, nodes])) +
                                           np.sum(np.abs(c_rel[node_id, nodes]))
                             for element_name, nodes in nodes_ids.items()}
                # find element maximally coupled to the current node

                max_coupling = max(couplings.items(), key=lambda x: x[1])
                if max_coupling[1] > coupling_threshold:
                    iteration_stalled = False
                    splitting_full_next[max_coupling[0]].append(node)
                    loose_nodes -= {node}

            splitting_full = splitting_full_next
            if iteration_stalled:
                # seems like a node that is impossible to excite anyway, so let's just get rid of it.
                # no error raised
                break
        return splitting_full

    def split(self, splitting: Mapping[str, Iterable[Any]],
              omitted_nodes: str = 'diagonal', coupling_threshold=1e-8) -> list:
        '''
        Split the linear element into several elements.
        :param splitting: dict where the keys are the names of the new split elements, and the values are iterables
        of terminals and internal degrees of freedom that will be assigned to that element
        :param omitted_nodes: strategy on how to handle nodes that are not explicitely
        listed in splitting. If set to 'diagonal', all nodes that haven't been featured in any splitting
        will be added to the 'diagonal' parts; if set to 'offdiagonal', all nodes that haven't been feature in any
        splitting will be added to the 'off-diagonal' parts; if set to 'tl_or_error', it will call the 'splitting_by_tl'
        attribute of the element if there is an omitted node. TODO: for now only 'diagonal' is implemented
        :param coupling_threshold: minimum relative coupling stregth (as defined by the capacitance or inverse
        inductance matrices) to be considered as non-zero.
        :return: list of SplitElementPart
        '''
        splitting_full = self.assign_loose_nodes(splitting, coupling_threshold=coupling_threshold)
        if omitted_nodes == 'diagonal':
            parts_diagonal = {(element_name,): SplitElementPart(element_name, self, nodes, nodes)
                                for element_name, nodes in splitting_full.items()}
            parts_offdiagonal = {(element_name1, element_name2):
                                     SplitElementPart(frozenset({element_name1, element_name2}), self, nodes1, nodes2)
                                     for eid1, (element_name1, nodes1) in enumerate(splitting_full.items())
                                     for eid2, (element_name2, nodes2) in enumerate(splitting_full.items())
                                     if eid1 < eid2}
        else:
            raise NotImplementedError(f'omitted_nodes strategy "{omitted_nodes}"')

        parts = dict()
        parts.update(parts_diagonal)
        parts.update(parts_offdiagonal)
        for part in parts.values():
            part.splitting_parts = parts
        return parts


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
    def __init__(self, name, element: CircuitElement,
                 terminals_a: Iterable, terminals_b: Iterable,
                 splitting_parts: Iterable = None):
        '''

        :param name: Name of the split element
        :param element: CircuitElement that is being split up
        :param terminals_a: Terminals of element that will be added to this element. If terminals are mentioned both in
        terminals_a and in terminals_b, the diagonal elements of the li, c and ri matrices will be appended. If a
        terminal is featured only in terminals_a or terminals_b, then only the off-diagonal elements between this
        terminal and those in terimnals_b or terminals_a will be added.
        :param terminals_b:
        :param splitting_parts: list of elements that belong to the same splitting, used to reconcilliation
        '''
        self.source = element
        self.terminals_a = terminals_a
        self.terminals_b = terminals_b
        self.name = name
        if splitting_parts is not None:
            self.splitting_parts = splitting_parts
        else:
            self.splitting_parts = []

    def get_element_licri(self):
        sli, sc, sri = self.source.get_element_licri()
        terminals = self.get_terminal_names()

        li = np.zeros((len(terminals), len(terminals)), sli.dtype)
        c = np.zeros((len(terminals), len(terminals)), sc.dtype)
        ri = np.zeros((len(terminals), len(terminals)), sri.dtype)

        source_terminal_indeces_a = [self.source.get_terminal_names().index(t) for t in self.terminals_a]
        source_terminal_indeces_b = [self.source.get_terminal_names().index(t) for t in self.terminals_b]
        source_terminal_indeces_a, source_terminal_indeces_b = \
            np.meshgrid(source_terminal_indeces_a, source_terminal_indeces_b)
        source_terminal_indeces_a = source_terminal_indeces_a.ravel()
        source_terminal_indeces_b = source_terminal_indeces_b.ravel()
        terminal_indeces_a = [terminals.index(t) for t in self.terminals_a]
        terminal_indeces_b = [terminals.index(t) for t in self.terminals_b]
        terminal_indeces_a, terminal_indeces_b = np.meshgrid(terminal_indeces_a, terminal_indeces_b)
        terminal_indeces_a = terminal_indeces_a.ravel()
        terminal_indeces_b = terminal_indeces_b.ravel()
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

