import numpy as np
from scipy.constants import h, hbar, e
from .circuit import Circuit
from .linear import CircuitElement, LinearElement
from typing import List


class _CircuitLoop:
    def __init__(self, elements: List[CircuitElement]):
        self.__elements = elements
        self.flux = 0.

    @property
    def elements(self):
        return self.__elements


class CircuitStationaryPhases:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.circuit_loops = dict()

        self.__node_names_stationary = []
        self.__connections_stationary = {}

        self.__constraint_indices = {}
        self.__li_stationary = np.empty(0)

        self.__stationary_phases = np.empty(0)

    @property
    def connections(self):
        return self.__connections_stationary

    @property
    def node_names(self):
        return self.__node_names_stationary

    @property
    def stationary_phases(self):
        return self.__stationary_phases

    def add_circuit_loop(self, loop_name: str, loop: List[CircuitElement], flux: float = 0.):
        """
        Add loop to a circuit
        :param loop_name: name of a loop
        :param loop: list of two-port CircuitElement instances that forms a loop
        :param flux: external flux in a loop in radians
        """
        if loop_name in self.circuit_loops.keys():
            raise ValueError(f"Loop {loop_name} is already presented in the circuit")
        loop = _CircuitLoop(elements=loop)
        self.__validate_loop(loop)
        self.circuit_loops[loop_name] = loop
        self.circuit_loops[loop_name].flux = flux

    def delete_loop(self, loop_name: str or None = None):
        """Delete loop or all loops presented in a circuit"""
        if loop_name is None:
            self.circuit_loops.clear()
        else:
            self.circuit_loops.pop(loop_name)

    def __validate_loop(self, loop: _CircuitLoop):
        for element in loop.elements:
            if element.name not in self.circuit.elements.keys():
                raise ValueError(f"Element with name {element.name} is not presented in the circuit")
            connections = self.circuit.connections[element.name]
            if len(connections.values()) != 2:
                raise ValueError(f"Element with name {element.name} is not a two-port element")

    def __add_stationary_nodes(self):
        """
        This private method adds a stationary node for each flux loop presented in a circuit
        """
        node_names, connections = self.circuit.connections_circuit()

        # names of node generalized coordinates including stationary nodes
        node_names_stationary = [node for node in node_names]
        for loop_name, loop in self.circuit_loops.items():
            element = loop.elements[0]
            connections = self.circuit.connections[element.name]
            element_reference_node = list(connections.values())[0]
            stationary_node = f"stationary_node_{loop_name}"
            print(
                f"Stationary node {stationary_node} for loop {loop_name} was added to the circuit {self.circuit.name}")
            node_names_stationary.append(stationary_node)

            index = node_names.index(element_reference_node) if element_reference_node in node_names else None
            # this two element tuple defines constraint for a loop
            constraint_indices = tuple([len(node_names_stationary) - 1, index])
            self.__constraint_indices[stationary_node] = constraint_indices
            self.__connections_stationary[element.name] = {'i': stationary_node, 'o': list(connections.values())[1]}

        element_mask = [element for element in self.circuit.elements.values() if
                        element.name not in list(self.__connections_stationary.keys())]
        li, *_ = self.circuit.get_system_licri(element_mask=element_mask, linear=True)

        li_stationary = np.zeros((len(node_names_stationary), len(node_names_stationary)), dtype=complex)
        li_stationary[:len(node_names), :len(node_names)] = li

        for element_name, connection in self.__connections_stationary.items():
            if not isinstance(self.circuit.elements[element_name], LinearElement):
                continue
            circuit_connection_ids = [node_names_stationary.index(node) if node in node_names_stationary else None for
                                      node in
                                      connection.values()]
            li_el, *_ = self.circuit.elements[element_name].get_element_licri()
            for i1, i2 in enumerate(circuit_connection_ids):
                if i2 is None:
                    continue
                for j1, j2 in enumerate(circuit_connection_ids):
                    if j2 is None:
                        continue
                    li_stationary[i2, j2] += li_el[i1, j1]

        for element_name, connection in self.circuit.connections.items():
            if element_name not in self.__connections_stationary:
                self.__connections_stationary[element_name] = connection
        self.__li_stationary = li_stationary
        self.__node_names_stationary = node_names_stationary

    def __get_element_phases(self, element_name: str, phases):
        """
        Get element phases from vector of system phases
        :param element_name: element name
        :param phases: phases in a circuit (including stationary nodes)
        """
        short_node_phase = 0.
        if element_name in self.__connections_stationary.keys():
            elem_nodes = self.__connections_stationary[element_name].values()
        else:
            elem_nodes = self.circuit.connections[element_name].values()
        circuit_connection_ids = [
            self.__node_names_stationary.index(node) if node in self.__node_names_stationary else None
            for node in elem_nodes]
        element_phases = np.asarray(
            [phases[connection] if connection is not None else short_node_phase for connection in
             circuit_connection_ids])
        return element_phases

    def __potential_energy(self, phases: np.ndarray):
        """
        Calculate potential energy as function of node phases with stationary nodes
        :param phases: vector of phases in radians
        """
        phi0 = hbar / 2 / e
        phases *= phi0

        # linear energy
        energy = 1 / 2 * phases.T.conj() @ self.__li_stationary @ phases

        # nonlinear energy
        for element_name, element in self.circuit.nonlinear_elements.items():
            element_phases = self.__get_element_phases(element_name, phases)
            energy += element.get_nonlinear_potential_energy(element_phases)
        scaling_factor = h
        return energy.real / scaling_factor

    def __get_stationary_phases_constraints(self):
        """This method returns stationary phases constraints"""
        constraints = []
        for loop_name, loop in self.circuit_loops.items():
            stationary_node = f"stationary_node_{loop_name}"
            index1, index2 = self.__constraint_indices[stationary_node]
            if index2 is not None:
                loop_constraint = {"type": "eq", "fun": lambda x: x[index1] - x[index2] - loop.flux}
            else:
                loop_constraint = {"type": "eq", "fun": lambda x: x[index1] - loop.flux}
            constraints.append(loop_constraint)
        return constraints

    def setup_stationary_phases(self, report=False):
        self.__add_stationary_nodes()
        if report:
            return self.__node_names_stationary, self.__constraint_indices, self.__li_stationary

    def get_stationary_phases(self):
        from scipy.optimize import minimize
        constraints = self.__get_stationary_phases_constraints()

        initial = np.zeros(len(self.__node_names_stationary))
        phase_bounds = (-np.pi, np.pi)
        bounds = [phase_bounds] * len(self.__node_names_stationary)

        # TODO: for SLSQP method add the gradient vector to minimization problem
        opt = minimize(self.__potential_energy, initial, method="SLSQP", bounds=None, constraints=constraints)
        self.__stationary_phases = opt.x
        return self.__stationary_phases

    def set_stationary_phases(self, stationary_phases, report=False):
        """Set stationary phases"""
        short_node_phase = 0.
        for element_name, element in self.circuit.elements.items():
            if hasattr(element, "stationary_phase"):
                element_phases = self.__get_element_phases(element_name, stationary_phases)
                element.stationary_phase = element_phases[1] - element_phases[0]
                if report:
                    print(f"Phase {element_phases} was set to element {element_name}")

    def reset_phases(self):
        """Set zero stationary phases to all elements"""
        for element_name, element in self.circuit.elements.items():
            if hasattr(element, "stationary_phase"):
                element.stationary_phase = 0.
