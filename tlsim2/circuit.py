import numpy as np
from abc import abstractmethod
from scipy.linalg import pinv, eig
from scipy.sparse.linalg import eigs
from typing import List, Mapping, Any, Iterable
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
                raise KeyError('No such connection in element: '+str(k))

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

        #rescaling modes
        v[v.shape[0]//2:] = v[v.shape[0]//2:]*np.sqrt(max_li/max_c)

        self.w = w*np.sqrt(max_li/max_c)
        self.v = v
        self.solution_valid = True

        return w*np.sqrt(max_li/max_c), v, node_names

    def get_response(self, omega: Iterable[float], element_name_in: Any, element_names_out: Iterable[Any],
                     terminals_in: Mapping[Any, float] = None,
                     terminals_out: Iterable[Mapping[Any, float]] = None,
                     z0_in: float=None, z0_out: Iterable[float]=None) -> np.ndarray:
        """
        Returns S-parameters when then excitation is applied to element_name_in and exits
        through element_names_out.
        :param omega: iterable of radial frequencies
        :param element_name_in: name of the input element (port)
        :param element_names_out: Iterable of names of the output elements (ports)
        :param terminals_in: dict where the keys are terminal names of the input element and
        the values are the weights with which they should be taken with. Defaults to
        {'i': +1, 'o': -1}, which is what you would have for a LumpedTwoTerminal.
        :param terminals_out: iterable of dicts where the keys are terminal names of
        the input elements and the values are the weights with which they should be
        taken with. Defaults to {'i': +1, 'o': -1} for each element_names_out
        :param z0_in: Impedance of input port for excitation calculation (so that there is
        no reflection, and also so the energy is normalized to unity). Defaults to element
        resistance (self.elements[element_name_in].r)
        :param z0_out: Impedances of the output ports for energy normalization (so
        that for ports with different impedances the S_21 parameter is normalized to unity).
        Defaults to element resistances (self.elements[element_name_out].r for element_name_out
        in element_names_out)
        :return: list of lists of complex s-parameters, for each output element, for each frequency.
        """
        if terminals_in is None:
            terminals_in = {'i': +1, 'o': -1}

        if terminals_out is None:
            terminals_out = [{'i': +1, 'o': -1} for i in range(len(element_names_out))]

        if z0_in is None:
            z0_in = self.elements[element_name_in].r

        if z0_out is None:
            z0_out = [self.elements[element_name_out].r for element_name_out in element_names_out]

        li, c, ri, node_names = self.get_system_licri()
        zeros = np.zeros_like(li)
        identity = np.identity(li.shape[0])

        b = np.hstack([
            np.vstack([zeros, identity]),
            np.vstack([c, zeros])
        ])
        a = np.hstack([
            np.vstack([-li, zeros]),
            np.vstack([ri, identity])
        ])

        voltage = np.sqrt(z0_in)
        current = -voltage / z0_in  # programming matched excitation

        ext_i = np.zeros(len(node_names)*2, dtype=complex)
        ext_v = np.zeros(len(node_names)*2, dtype=complex)

        nodes = {self.connections[element_name_in][t]: w for t, w in terminals_in.items()
                 if self.connections[element_name_in][t] not in self.shorted_nodes}

        for node, weight in nodes.items():
            ext_i[node_names.index(node)] = weight * current
            ext_i[node_names.index(node) + len(node_names)] = weight * voltage
            ext_v[node_names.index(node) + len(node_names)] = weight * voltage

        m = b * 1j * np.reshape(omega, (-1, 1, 1)) + a
        ext = ext_i.reshape((1, -1, 1)) + \
              (b @ ext_v).reshape((1, -1, 1)) * 1j * np.reshape(omega, (-1, 1, 1))
        response = np.linalg.solve(m, ext)
        response = response.reshape(response.shape[:2])

        if not hasattr(element_names_out, '__iter__'):
            element_names_out = [element_names_out]

        voltage_responses = np.zeros((len(omega), len(element_names_out)), dtype=complex)

        for i, (element_terminals_out, element_name_out, element_z0_out) in enumerate(
                zip(terminals_out, element_names_out, z0_out)):
            nodes = {self.connections[element_name_out][t]: w for t, w in element_terminals_out.items()
                     if self.connections[element_name_out][t] not in self.shorted_nodes}

            voltage_responses[:, i] = (np.sum(
                [response[:, node_names.index(node) + len(node_names)] for node, weight in nodes.items()],
                axis=0) / np.sqrt(element_z0_out))

        return voltage_responses

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

    def element_epr(self, elements, modes=None, return_losses=False):
        """
        Return mode ids in order of participation of elements
        :param elements: elements to search for modes
        :return:
        """
        element_energies = []
        element_losses = []
        total_energies = []
        li_el, c_el, ri_el, node_names = self.get_system_licri(elements)
        li, c, ri, node_names = self.get_system_licri()

        if modes is None:
            modes = self.v

        for mode_id in range(modes.shape[1]):
            voltages = modes[modes.shape[0]//2:, mode_id]
            phases = modes[:modes.shape[0]//2, mode_id]
            element_energies.append(np.conj(phases).T@li_el@phases + np.conj(voltages).T@c_el@voltages)
            element_losses.append(np.conj(voltages).T@ri_el@voltages)
            total_energies.append(np.conj(phases).T@li@phases + np.conj(voltages).T@c@voltages)

        epr = np.asarray(element_energies)/np.asarray(total_energies)
        losses = np.asarray(element_losses)/np.asarray(total_energies)

        print(total_energies, element_energies, element_losses)

        if not return_losses:
            return epr
        else:
            return epr, losses

    def make_element(self, nodes: Mapping[Any, Any], cutoff_low: float = 1e3, cutoff_high: float = 1e11):
        from .subcircuit import Subcircuit
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

    def autosplit(self, keep_nodes=list(), cutoff_low=1e3, cutoff_high=1e11):
        """
        Create new circuit consisting of 'subcircuits', breaking apart according to coupling_hints of elements.
        :return: Circuit consisting of Circuits with the same elements as the parent, except for the elements that
        are split by according to coupling_hints.
        """
        connections_split = []

        for element_name, element in self.elements.items():
            connections = self.connections[element_name]
            coupled_nodes = element.get_coupling_hints()

            for terminal1, node1 in connections.items():
                if node1 in self.shorted_nodes:
                    continue
                for terminal2, node2 in connections.items():
                    if node2 in self.shorted_nodes:
                        continue
                    if node1 == node2:
                        continue
                    for subsystem in coupled_nodes:
                        if terminal1 in subsystem and terminal2 in subsystem:
                            connections_split.append({node1, node2})

        # go bfs
        subgraphs = []
        disconnected_nodes = list(
            set([v for n in self.connections.values() for v in n.values() if v not in self.shorted_nodes]))
        while len(disconnected_nodes) > 0:
            subgraph = {disconnected_nodes[0]}
            node_stack = [disconnected_nodes[0]]
            while len(node_stack) > 0:
                node = node_stack.pop()
                if node in disconnected_nodes:
                    disconnected_nodes.remove(node)
                else:
                    continue
                for connection in connections_split:
                    if node in connection:
                        node_stack.extend(connection)
                        subgraph = subgraph | connection
            if len(subgraph) > 0:
                subgraphs.append(frozenset(subgraph))

        # split subsystems
        subsystems = {subgraph: Circuit(name=(self.name, subgraph)) for subgraph in subgraphs}
        split_system = Circuit(name=(self.name, 'split'))

        circuit_nodes, connections_circuit = self.connections_circuit()

        for element_name, element in self.elements.items():
            connections = self.connections[element_name]

            splitting = {}
            for subsystem in subgraphs:
                part = subsystem & set(connections.values())
                part_keys = [k for k, v in connections.items() if v in part]
                if len(part_keys) > 0:
                    splitting[(element_name, subsystem)] = part_keys
            if len(splitting) < 2:
                split_elements = {(k,): element for k in splitting.keys()}
            else:
                split_elements = element.split(splitting=splitting)
            #         print(split_elements)

            for k, splitted_element in split_elements.items():
                connections_splitted_element = {k2: v for k2, v in connections_circuit[element_name].items()
                                                if k2 in splitted_element.get_terminal_names()}
                keep_top_level = (len(k) > 1)
                if hasattr(splitted_element, 'keep_top_level'):
                    if splitted_element.keep_top_level:
                        keep_top_level = True
                if not keep_top_level:
                    #                 system = subsystems[frozenset(connections_splitted_element.values())]
                    system = subsystems[k[0][1]]
                else:
                    system = split_system
                system.add_element(splitted_element, connections_splitted_element)

        split_system_nodes = split_system.connections_circuit()[0]
        for subgraph, subsystem in subsystems.items():
            for s in self.shorted_nodes:
                if s in subsystem.connections_circuit()[0]:
                    subsystem.short(s)
            # need to find which nodes are outward-facing
            connections = {n: n for n in subsystem.connections_circuit()[0] if n in split_system_nodes or n in keep_nodes}
            split_system.add_element(subsystem.make_element(connections, cutoff_low=cutoff_low,
                                                            cutoff_high=cutoff_high), connections)

        for s in self.shorted_nodes:
            if s in split_system.connections_circuit()[0]:
                split_system.short(s)

        return split_system

    def find_subsystem_name_by_element(self, element):
        for name, subsystem_or_element in self.elements.items():
            if hasattr(subsystem_or_element, 'circuit'):
                if (element in subsystem_or_element.circuit.elements.values()
                        or element in subsystem_or_element.circuit.elements.keys()):
                    return name
        return None

    def rename_element(self, old_name, new_name):
        self.connections[new_name] = self.connections[old_name]
        self.elements[new_name] = self.elements[old_name]
        self.elements[new_name].name = new_name
        del self.connections[old_name]
        del self.elements[old_name]

