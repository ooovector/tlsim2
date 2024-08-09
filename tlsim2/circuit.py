import numpy as np
from abc import abstractmethod
from scipy.linalg import pinv, eig
from scipy.sparse.linalg import eigs
from scipy.optimize import fsolve
from typing import List, Mapping, Any, Iterable
from collections import OrderedDict
from scipy.constants import h, hbar, e
from matplotlib import pyplot as plt
from .linear import NonlinearElement


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

        self.li = None
        self.c = None
        self.ri = None
        self.ci = None

        self.nonlinear_elements = OrderedDict()

        self.node_names = []
        self.linear_coordinate_transform = np.asarray(0)
        self.variables_types = []

        self.variables = []
        self.phase_periods = 7  # phase periods for extended variables
        self.nodeNo = 128

    def short(self, node):
        self.shorted_nodes.append(node)

    def add_element(self, element, connection):
        if element.name in self.elements:
            raise ValueError(f'Element with name {element.name} already in Circuit')
        self.elements[element.name] = element

        if hasattr(element, 'get_lagrangian_series'):
            self.nonlinear_elements[element.name] = element
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
        Returns inverse inductance (li), capacitance (c) and resistance matrices (r) of system, including only
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

        # avoid NaNs
        if not max_li > 0:
            max_li = 1
        if not max_c > 0:
            max_c = 1

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
            # remove redundancy in modes
            w = np.imag(self.w)
            modes = self.v
            modes = modes[:, w > 0]

        for mode_id in range(modes.shape[1]):
            voltages = modes[modes.shape[0]//2:, mode_id]
            phases = modes[:modes.shape[0]//2, mode_id]
            element_energies.append(np.conj(phases).T@li_el@phases + np.conj(voltages).T@c_el@voltages)
            element_losses.append(np.conj(voltages).T@ri_el@voltages)
            total_energies.append(np.conj(phases).T@li@phases + np.conj(voltages).T@c@voltages)

        epr = np.asarray(element_energies)/np.asarray(total_energies)
        losses = 2*np.asarray(element_losses)/np.asarray(total_energies)

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

        return Subcircuit(self, modes=(cutoff_low, cutoff_high), nodes=nodes)

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

    def element_node_mapping(self, element):
        input_nodes, output_nodes = [], []
        for terminal in self.connections[element.name]:
            if terminal[0] == 'i':
                input_nodes.append(self.connections[element.name][terminal])
            else:
                output_nodes.append(self.connections[element.name][terminal])
        return input_nodes, output_nodes

    def flux_connections_matrix(self, flux_zero_elements):
        """
        Returns matrix A of flux connection equations in form A \vec{Phi} = 0, where \vec{Phi} is a node fluxes vector
        """
        node_names, connections = self.connections_circuit()
        constraint_equations = []
        for element in self.elements.values():
            flux_equation = np.zeros((len(node_names)))
            if element.type_ in flux_zero_elements:
                input_nodes, output_nodes = self.element_node_mapping(element)
                input_nodes_ids = [node_names.index(node) if node in node_names else None \
                                   for node in input_nodes]
                output_nodes_ids = [node_names.index(node) if node in node_names else None \
                                    for node in output_nodes]
                if element.lumped:
                    if input_nodes_ids[0] is not None:
                        flux_equation[input_nodes_ids[0]] = 1
                    if output_nodes_ids[0] is not None:
                        flux_equation[output_nodes_ids[0]] = -1
                    constraint_equations.append(flux_equation)
                else:
                    raise NotImplementedError
            if element.type_ == 'TL':
                # we consider multi conductor transmission line
                input_nodes, output_nodes = self.element_node_mapping(element)
                input_nodes_ids = [node_names.index(node) if node in node_names else None \
                                   for node in input_nodes]
                output_nodes_ids = [node_names.index(node) if node in node_names else None \
                                    for node in output_nodes]
                if 'L' in flux_zero_elements:
                    for id_, input_node_id in enumerate(input_nodes_ids):
                        if input_node_id is not None:
                            flux_equation[input_nodes_ids[0]] = 1
                        if output_nodes_ids[id_] is not None:
                            flux_equation[output_nodes_ids[id_]] = -1
                        constraint_equations.append(flux_equation)

        # print(constraint_equations)
        if not len(constraint_equations):
            constraint_equations = [np.zeros(node_names)]
        flux_connections_matrix = np.zeros((len(constraint_equations), len(node_names)))
        for i, v in enumerate(constraint_equations):
            flux_connections_matrix[i, :] = v
        # print(flux_connections_matrix)
        return flux_connections_matrix

    @staticmethod
    def gauss_jordan_elimination(mat, full_output=False):
        """
        The method provides Gauss-Jordan elimination for matrix in the form (A|b)
        to find fundamental system of solutions
        :param mat: matrix in the form (A|b)
        :param full_output: if True, when returns ordering parameters
        """

        def p_ij_mapping(mat, i, j):
            """
            Swap two rows
            """
            mat[[i, j]] = mat[[j, i]]

        def d_i_lamda(mat, i, lamda):
            """
            Multiply row by lamda
            """
            mat[i] = np.asarray([a * lamda for a in mat[i]])

        def t_ij_lamda(mat, i, j, lamda):
            """
            Add to row i another multiplied row
            """
            mat[i] = np.asarray([(a + b * lamda) for a, b in zip(mat[i], mat[j])])

        count = 0
        order = []
        n = mat.shape[1] - 1  # number of variables
        m = mat.shape[1]  # number of equations
        r = 0
        for j in range(mat.shape[1]):
            column = mat[count:, j]
            if np.all(column == 0):
                continue
            order.append(j)
            r += 1
            for i in range(count, mat.shape[0]):
                # find pivot element in this column
                aij = mat[i, j]
                if aij != 0:
                    p_ij_mapping(mat, count, i)
                    d_i_lamda(mat, count, 1 / aij)
                    break
            for k in range(mat.shape[0]):
                if k != count:
                    a_kj = mat[k, j]
                    t_ij_lamda(mat, k, count, -a_kj)
            count += 1
        # print(order)
        # make new order of x vector
        for indx in range(mat.shape[1]):
            if indx not in order:
                order.append(indx)
        mat = mat[:, order]
        # print(r)
        # Er = mat[:r, :r] # eye matrix
        Phi = mat[:r, r:-1]
        b = mat[:r, -1]
        fss_mat = np.vstack((-Phi, np.eye(Phi.shape[1])))
        b = np.hstack((b, np.zeros(m - r)))
        # print(order)
        # reorder for the first variables
        order_in = [i for i in range(m)]
        reorder = [order.index(order_in[i]) for i in range(m)]
        mat = mat[:, reorder]
        fss_mat = fss_mat[reorder[:-1], :]
        b = b[reorder[:-1]]
        if full_output:
            return mat, fss_mat, b, (r, order, reorder)
        else:
            return mat, fss_mat, b

    def define_subspaces(self, subspace_type):
        """
        This method defines free, frozen or periodic subspaces.
        Free subspace is corresponding to free particles (without potential). Free variables are generated by isolated
        islands formed by capacitors.
        Frozen variables are corresponding to massless particles with potential energy.
        Periodic variables formed by superconducting islands where charge tunneling is possible.
        """
        if subspace_type not in ['free', 'frozen', 'periodic']:
            raise ValueError('Subspace type should be free, frozen or periodic!')
        flux_zero_elements = {'free': ['L', 'JJ', 'R'], 'frozen': ['C'], 'periodic': ['L', 'R']}
        flux_connections_matrix = self.flux_connections_matrix(flux_zero_elements[subspace_type])
        free_column = np.zeros(flux_connections_matrix.shape[0]).reshape(flux_connections_matrix.shape[0], 1)
        mat = np.hstack((flux_connections_matrix, free_column))
        mat, fss_mat, b = self.gauss_jordan_elimination(mat)
        return mat, fss_mat, b

    def get_variable_transformation(self):
        """
        Create transformation matrix T for free, frozen, periodic and extended variables:
        φ = Tθ, where T is a square invertible  matrix.
        """
        node_names, connections = self.connections_circuit()
        _, fss_frozen, _ = self.define_subspaces('frozen')
        _, fss_free, _ = self.define_subspaces('free')
        _, fss_periodic, _ = self.define_subspaces('periodic')
        fss_types = ['frozen'] * fss_frozen.shape[1] + ['free'] * fss_free.shape[1] + ['periodic'] * fss_periodic.shape[1]
        basis = np.hstack((fss_free, fss_frozen, fss_periodic))
        # print(basis, fss_types)
        # check linear independency of vector set
        mat = np.hstack((basis, np.zeros((basis.shape[0], 1))))
        mat, fss_mat, b, params_ = self.gauss_jordan_elimination(mat, full_output=True)
        rank = params_[0]
        order = params_[1][:rank]
        # print(rank, order)
        basis = basis[:, order]
        fss_types = [fss_types[i] for i in order]
        if basis.shape[1] < len(node_names):
            # add basis vectors to make transformation matrix invertible (find kernel)
            mat = np.hstack((basis.T, np.zeros((basis.shape[1], 1))))
            mat, kernel, b = self.gauss_jordan_elimination(mat)
            basis = np.hstack((basis, kernel))
            fss_types += ['extended' for i in range(kernel.shape[1])]
        self.linear_coordinate_transform = basis
        self.variables_types = fss_types
        return basis, fss_types

    def create_quantum_circuit(self):
        """
        Create quantum circuit
        """
        # make variables transformation
        li, c, ri, node_names = self.get_system_licri()
        c_new = np.einsum('ji,jk,kl->il', self.linear_coordinate_transform, c, self.linear_coordinate_transform)
        li_new = np.einsum('ji,jk,kl->il', self.linear_coordinate_transform, li, self.linear_coordinate_transform)
        # eliminate frozen variables (because the capacitance matrix is singular in the presence of them)
        # and free variables
        indx = len([i for i in self.variables_types if i in ['frozen', 'free']])
        self.li, self.ci = li_new[indx:, indx:], np.linalg.inv(c_new[indx:, indx:])

        # create quantum variables
        variables_types = self.variables_types[indx:]
        self.variables = []  # here we add only periodic and extended variables
        for i, v in enumerate(variables_types):
            from .qvariable import QVariable
            variable = QVariable('θ_{}_{}'.format(i, v))
            phase_periods = 1 if v == 'periodic' else self.phase_periods
            variable.create_grid(self.nodeNo, phase_periods)
            self.variables.append(variable)

    def grid_shape(self):
        return tuple([v.get_nodeNo() for v in self.variables])

    def create_phase_grid(self):
        """
        Creates a n-d grid of the phase variables, where n is the number of variables in the circuit,
        on which the circuit wavefunction depends.
        """
        axes = []
        for variable in self.variables:
            axes.append(variable.get_phase_grid())
        return np.meshgrid(*tuple(axes), indexing='ij')

    def create_charge_grid(self):
        """
        Creates a n-d grid of the charge variables, where n is the number of variables in the circuit,
        on which the circuit wavefunction, when transformed into charge representation, depends.
        """
        axes = []
        for variable in self.variables:
            axes.append(variable.get_charge_grid())
        return np.meshgrid(*tuple(axes), indexing='ij')

    def calculate_charge_potential(self):
        grid_shape = self.grid_shape()
        grid_size = np.prod(grid_shape)
        charge_grid = np.reshape(np.asarray(self.create_charge_grid()), (len(self.variables), grid_size))
        ecmat = 2 * e ** 2 * self.ci
        self.charge_potential = np.einsum('ij,ik,kj->j', charge_grid, ecmat, charge_grid)
        self.charge_potential = np.reshape(self.charge_potential, grid_shape)
        return self.charge_potential

    def calculate_phase_potential(self):
        grid_shape = self.grid_shape()
        grid_size = np.prod(grid_shape)
        phase_grid = np.reshape(np.asarray(self.create_phase_grid()), (len(self.variables), grid_size))
        Phi0 = h / 2 / e
        elmat = 1 / 2 * Phi0 ** 2 * self.li / (2 * np.pi) ** 2
        # liner part of phase potential
        self.phase_potential = np.einsum('ij,ik,kj->j', phase_grid, elmat, phase_grid)

        node_names, connections = self.connections_circuit()
        indx = len([i for i in self.variables_types if i in ['frozen', 'free']])
        node_phases = np.einsum('ij,jk->ik', self.linear_coordinate_transform[:, indx:], phase_grid)
        # nonlinear part of phase potential
        for element in self.elements.values():
            if hasattr(element, 'nonlinear_energy_term'):
                input_nodes, output_nodes = self.element_node_mapping(element)
                nodes_indx = [node_names.index(node) if node in node_names else None for node in
                              input_nodes + output_nodes]
                element_node_phases = [node_phases[i] if i is not None else np.zeros(node_phases.shape[1]) for i in
                                       nodes_indx]
                self.phase_potential += element.nonlinear_energy_term(element_node_phases)
        self.phase_potential = np.reshape(self.phase_potential, grid_shape)
        return self.phase_potential

    def hamiltonian_phase_action(self, state_vector):
        """
        Implements the action of the hamiltonian on the state vector describing the system in phase representation.
        :param state_vector: wavefunction to act upon
        :returns: wavefunction after action of the hamiltonian
        """
        psi = np.reshape(state_vector, self.charge_potential.shape)  # wavefunction in phase representation
        phi = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(psi)))  # wavefunction in charge representation
        Up = self.phase_potential.ravel()*state_vector
        Tp = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(self.charge_potential*phi))).ravel()
        return Tp + Up

    def calculate_potentials(self):
        """
        Calculate potentials for Fourier-based hamiltonian action.
        """
        self.calculate_phase_potential()
        self.calculate_charge_potential()
        from scipy.sparse.linalg import LinearOperator
        self.hamiltonian = LinearOperator((np.prod(self.grid_shape()), np.prod(self.grid_shape())),
                                          matvec=self.hamiltonian_phase_action)
        return self.charge_potential, self.phase_potential

    def optimize_nodes(self, w_ideal, decay_ideal, param_dict, mode_elements):
        """
        Optimizes eigenfrequencies and decay constants to the values given
        in w_ideal and decay_ideal by changing parameters specified in
        param_dict.
        The modes which eigenfrequencies and decay constants are optimized
        are determined by having the highest energy dissipation ratio at the
        elements given in mode_elements.
        :param w_ideal: optimized eigenfrequencies in GHz
        :param decay_ideal: optimized decay constants in GHz
        :param param_dict param_dict: dictionary of parameters which are
        changed to achieve ideal eigenmodes
        :param mode_elements: list of elements from the circuits which
        are used to identify the correct modes
        :return: returns the optimization result (frequencies and decay
        constants of last step of optimization)
        """
        def cost_fct(values, elem_dict, w_ideal, decay_ideal, mode_elements):
            for (elem, par), new_value in zip(elem_dict.items(), values):
                new_value = 1e-6 if new_value <= 0 else new_value
                if par == 'c':
                    new_value = new_value / 1e15

                setattr(self.elements[elem], par, new_value)
            w_rpf, v_rpf, node_names_rpf = self.compute_system_modes()
            frequencies = np.imag(w_rpf / (2 * np.pi)) / 1e9
            decays = np.real(w_rpf / 1e9) * 2
            cost = []

            indices = self.find_mode_of_elements(
                mode_elements, n=len(w_ideal))

            for i, j in enumerate(indices):
                cost.append(frequencies[j] - w_ideal[i])
                cost.append(decays[j] - decay_ideal[i])
            return cost

        x0 = []
        for elem, par in param_dict.items():
            if par == 'c':
                x0.append(getattr(self.elements[elem], par) * 1e15)
            else:
                x0.append(getattr(self.elements[elem], par))

        return fsolve(cost_fct, x0, args=(param_dict, w_ideal, decay_ideal,
                                          mode_elements), )

    def find_mode_of_elements(self, elements, only_positives=True, n=2,
                              threshold=0.5):
        """
        Returns the mode indices of the n lowest modes which have the maximum
        energy participation ration at the elements.
        :param elements: list of elements which energy participation ration
        is calculated
        :param only_positives: if True, only the positive frequencies are
        considered
        :param n: number of returned mode indices
        :return: list of mode indices
        """
        modes = np.abs(self.element_epr(elements)) > threshold
        freqs = np.imag(self.w)
        if only_positives:
            positive_frequencies_mask = freqs > 0
        else:
            positive_frequencies_mask = np.ones_like(modes, dtype=bool)
        mask = np.logical_and(modes, positive_frequencies_mask)
        freqs[np.logical_not(mask)] = np.inf
        relevant_modes = np.argsort(freqs)[:n]
        return relevant_modes


