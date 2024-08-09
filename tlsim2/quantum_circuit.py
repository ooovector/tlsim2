from .circuit import Circuit
from .linear import NonlinearElement
import numpy as np
from scipy.constants import hbar


class QuantumCircuit:
    def __init__(self, linear_circuit: Circuit):
        self.circuit = linear_circuit

    @property
    def num_modes(self):
        """Number of modes M in the system"""
        return len(self.circuit.w) // 2

    @property
    def num_nonlinear_elements(self):
        """Number of nonlinear elements presented in the system"""
        return len(self.circuit.nonlinear_elements.items())

    def get_epr_coefficients(self):
        """
        Return matrix of EPR coefficients with shape MxJ, where M is number of modes in the system and J is a number
        of nonlinear elements
        """
        # remove redundancy in modes
        w = np.imag(self.circuit.w)
        modes = self.circuit.v
        modes = modes[:, w > 0]

        epr_mat = np.empty((self.num_modes, self.num_nonlinear_elements))
        for j, nonlinear_element in enumerate(self.circuit.nonlinear_elements.values()):
            element_epr = self.circuit.element_epr([nonlinear_element], modes=modes)
            epr_mat[:, j] = np.real(element_epr) * 2
        return epr_mat

    def get_kerr_matrix(self):
        epr_mat = self.get_epr_coefficients()
        w = np.imag(self.circuit.w)
        w = w[w > 0]
        w_mat = np.diag(w)
        ej_mat = np.diag([element.ej for element in self.circuit.nonlinear_elements.values()])
        temp = w_mat @ epr_mat
        kerr_matrix = hbar / 4 * temp @ np.linalg.inv(ej_mat) @ temp.T
        anharmonicities = np.diag(kerr_matrix) / 2
        return kerr_matrix, anharmonicities

    def get_hamiltonian_perturbation(self):
        raise NotImplementedError

    @staticmethod
    def create(n: int):
        """Returns matrix representation of an n-dimensional creation operator"""
        diag = np.sqrt(np.arange(1, n))
        mat = np.zeros([n, n])
        np.fill_diagonal(mat[1:], diag)
        return mat

    @staticmethod
    def destroy(n: int):
        """Returns matrix representation of an n-dimensional annihilation operator"""
        diag = np.sqrt(np.arange(1, n))
        mat = np.zeros([n, n])
        np.fill_diagonal(mat[:, 1:], diag)
        return mat





