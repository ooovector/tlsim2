import numpy as np
from abc import abstractmethod
from scipy.linalg import pinv, eig
from scipy.sparse.linalg import eigs
from typing import List, Mapping, Any
from collections import OrderedDict

from matplotlib import pyplot as plt


class Circuit:
    def __init__(self, name=None):
        self.name = name
        self.node_names = []
        self.elements = OrderedDict()
        self.connections = dict()
        self.shorted_nodes = []

        self.terminal_nodes = {}

        self.solution_valid = False
        self.w = None
        self.v = None

    def short(self, node):
        self.shorted_nodes.append(node)

    def add_element(self, element, connection):
        if element.name in self.elements:
            raise ValueError(f'Element with name {element.name} already in Circuit')
        self.elements[element.name] = element
        self.connections[element.name] = connection

        for k, v in connection.items():
            if k not in element.get_terminal_names():
                raise KeyError('No such connection in element')

        for node_name in connection.values():
            if node_name not in self.node_names:
                self.node_names.append(node_name)

        self.solution_valid = False

    def get_num_dof(self):
        return len(self.node_names)

    def connections_circuit(self):
        node_names = [name for name in self.node_names if name not in self.shorted_nodes]
        connections = {element_name: {k: v for k, v in connection.items()} \
                            for element_name, connection in self.connections.items()}
        # for element_id, element, connection in zip(range(len(self.elements)), self.elements, connections):
        for element in self.elements.values():
            connection = connections[element.name]
            element_nodes = element.get_terminal_names()
            for node_name in element_nodes:
                if node_name not in connection:
                    new_node_name = f'{element.name}_{node_name}_disconnected'
                    connection[node_name] = new_node_name
                    node_names.append(new_node_name)
        return node_names, connections

    def get_system_licri(self, element_mask: List = None):
        """
        Returns inverse inductance (li), capcacitance (c) and resistance matrices (r) of system, including only
        contributions by circuit elements in element_mask (all elements, if element_mask is None)
        :param element_mask:
        :return:
        """
        node_names, connections = self.connections_circuit()

        li = np.zeros((len(node_names), len(node_names)), dtype=np.complex128)
        c = np.zeros((len(node_names), len(node_names)), dtype=np.complex128)
        ri = np.zeros((len(node_names), len(node_names)), dtype=np.complex128)

        if element_mask is None:
            element_mask = [element for element in self.elements.values()]

        for element in self.elements.values():
            connection = connections[element.name]

            if element not in element_mask:
                continue
            circuit_connection_ids = [node_names.index(connection[t]) if connection[t] in node_names else None\
                                      for t in element.get_terminal_names()]
            li_el, c_el, ri_el = element.get_element_licri()
            for i1, i2 in enumerate(circuit_connection_ids):
                if i2 is None:
                    continue
                for j1, j2 in enumerate(circuit_connection_ids):
                    if j2 is None:
                        continue
                    li[i2, j2] += li_el[i1, j1]
                    c[i2, j2] += c_el[i1, j1]
                    ri[i2, j2] += ri_el[i1, j1]

        return li, c, ri, node_names

    def get_terminal_names(self):
        return [k for k in self.terminal_nodes.keys()]

    def get_terminal_modes(self):
        return {terminal: np.asarray([1 if node == current_node else 0 for current_node in self.node_names]) \
                for terminal, node in self.terminal_nodes.items()}

    def compute_system_modes(self, num_modes=None):
        li, c, ri, node_names = self.get_system_licri()
        zeros = np.zeros_like(li)
        identity = np.identity(li.shape[0])

        # normalize matrices
        max_li = np.max(np.abs(li))
        max_c = np.max(np.abs(c))

        li_norm = li/max_li
        c_norm = c/max_c

        ri_norm = ri/np.sqrt(max_li*max_c)

        b = np.hstack([
            np.vstack([zeros, identity]),
            np.vstack([c_norm, zeros])
        ])
        a = np.hstack([
            np.vstack([-li_norm, zeros]),
            np.vstack([ri_norm, identity])
        ])

        if num_modes is None or num_modes >= a.shape[0] - 1:
            w, v = eig(a, b)
        else:
            w, v = eigs(a, k=num_modes, M=b, which='SM')

        self.w = w*np.sqrt(max_li/max_c)
        self.v = v
        self.solution_valid = True

        return w*np.sqrt(max_li/max_c), v, node_names

    def system_modes_to_element_modes(self, system_modes, elements):
        node_names, circuit_connections = self.connections_circuit()

        element_modes = []

        for system_mode_id, system_mode in enumerate(system_modes.T):
            m = []
            for element in elements:
                element_name = element.name
                element_connections = circuit_connections[element_name]
                mode_fields = {}
                for element_terminal, node in element_connections.items():
                    if node not in node_names:
                        value = 0
                    else:
                        value = system_mode[node_names.index(node)]
                    mode_fields[element_terminal] = value
                m.append(mode_fields)
            element_modes.append(m)

        return element_modes

    def element_epr(self, elements, modes=None):
        """
        Return mode ids in order of participation of elements
        :param elements: elements to search for modes
        :return:
        """
        element_energies = []
        total_energies = []
        li_el, c_el, ri_el, node_names = self.get_system_licri(elements)
        li, c, ri, node_names = self.get_system_licri()

        if modes is None:
            modes = self.v

        for mode_id in range(modes.shape[1]):
            voltages = modes[modes.shape[0]//2:, mode_id]
            phases = modes[:modes.shape[0]//2, mode_id]
            element_energies.append(np.conj(phases).T@li_el@phases + np.conj(voltages).T@c_el@voltages)
            total_energies.append(np.conj(phases).T@li@phases + np.conj(voltages).T@c@voltages)

        return np.asarray(element_energies)/np.asarray(total_energies)

    def make_element(self, nodes: Mapping[Any, Any], cutoff_low: float = 1e3, cutoff_high: float = 1e11):
        from subcircuit import Subcircuit
        """
        Returns a Subcircuit element based on this circuit. Used for building larger circuits from smaller elements.
        :param nodes: dict of circuit nodes to new element modes
        :param cutoff_low: modes with frequency lower than this threshold will not be added into the internal modes
        :param cutoff_hig: modes with frequency higher than this threshold will not be added into the internal modes
        :return: Subcircuit [see Subcircuit documentation for more details]
        """
        if self.w is None:
            self.compute_system_modes()

        mode_mask = np.logical_and(np.imag(self.w) >= cutoff_low, np.imag(self.w) <= cutoff_high)

        modes = self.v[:self.v.shape[0]//2, :]
        modes = modes[:, mode_mask]
        modes = {mode_id: modes[:, mode_id] for mode_id in range(modes.shape[1])}

        return Subcircuit(self, modes=modes, nodes=nodes)
