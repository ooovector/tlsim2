import numpy as np
from abc import abstractmethod
from scipy.linalg import pinv, eig
from scipy.sparse.linalg import eigs
from .circuit import Circuit
from typing import Mapping, Any, Iterable
from collections import OrderedDict


class Subcircuit:
    def __init__(self, circuit: Circuit, nodes: Mapping[Any, Any], modes: Mapping[Any, Any],
                 keep_top_level=False):
        """
        Create a Subcircuit CircuitElement from a Circuit.
        :param circuit: Circuit from which the li, c, ri are taken from.
        :param nodes: Nodes that are retained as connectable in the CircuitElement
        :param modes: Internal modes that will be retained in the subcircuit
        :param keep_top_level: Flag that will be acknowledged by autosplit, leaving
        the element in the top-level system.
        """
        self.circuit = circuit
        self.nodes = nodes
        self.modes = modes
        self.name = self.circuit.name
        # self.cutoff_low = cutoff_low
        # self.cutoff_high = cutoff_high

        self.li = np.zeros((0, 0))
        self.c = np.zeros((0, 0))
        self.ri = np.zeros((0, 0))
        self.node_names_el = None

        self.compute_element_licri()

    def compute_element_licri(self, residual_threshold=1e-8):
        li_sys, c_sys, ri_sys, node_names_sys = self.circuit.get_system_licri()

        modes_with_nodes = {node_element: np.asarray([1 if node_id == node_names_sys.index(node_system) else 0 \
                                                for node_id in range(len(node_names_sys))]) \
                            for node_system, node_element in self.nodes.items()}

        node_mask = np.ones(len(node_names_sys))
        system_nodes, system_connections = self.circuit.connections_circuit()
        for node_system, node_element in self.nodes.items():
            node_mask[system_nodes.index(node_system)] = 0

        for mode_name, mode in self.modes.items():
            mode_no_nodes = mode * node_mask
            rank = np.linalg.matrix_rank([[mode2 for mode2 in modes_with_nodes.values()] + [mode]],
                                         tol=residual_threshold)

            if rank > len(modes_with_nodes):
                modes_with_nodes[('mode', mode_name)] = mode_no_nodes

        modes_with_nodes = OrderedDict(modes_with_nodes)
        # modes complete. This is now our transform

        node_names_el = []
        modes = []

        for node_id, node_name in enumerate(modes_with_nodes):
            node_names_el.append(node_name)
            modes.append(modes_with_nodes[node_name])

        li_el = np.einsum('ij,jk,lk->il', np.conj(modes), li_sys, modes)
        c_el = np.einsum('ij,jk,lk->il', np.conj(modes), c_sys, modes)
        ri_el = np.einsum('ij,jk,lk->il', np.conj(modes), ri_sys, modes)

        # symmetrize (remove rounding errors)
        # self.li = (li_el + li_el.conj().T)/2
        # self.c = (c_el + c_el.conj().T)/2
        # self.ri = (ri_el + ri_el.conj().T)/2

        self.li = li_el
        self.c = c_el
        self.ri = ri_el
        self.node_names_el = node_names_el

    def get_terminal_names(self):
        return self.node_names_el

    def get_element_licri(self):
        return self.li, self.c, self.ri

    def get_coupling_hints(self):
        return [self.get_terminal_names()]
