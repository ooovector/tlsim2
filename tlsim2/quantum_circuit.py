from typing import List, Tuple
from .circuit import Circuit
from .linear import NonlinearElement
import numpy as np
from scipy.constants import hbar, e

PHI0 = hbar / (2 * e)


class QuantumCircuit:
    """
    This class provides second order quantization of a circuit in the basis of harmonic modes

    Attributes:
        circuit (Circuit): the linear circuit object

    Methods:
        get_epr_coefficients(): returns matrix of EPR coefficients
        get_kerr_matrix(): returns matrix of Kerr coefficients and anharmonicities followed by idea from
        https://www.nature.com/articles/s41534-021-00461-8
    """
    def __init__(self, linear_circuit: Circuit):
        self.circuit = linear_circuit

    @property
    def num_modes(self):
        """Number of modes M in the system"""
        return len(self.circuit.w)

    @property
    def num_nonlinear_elements(self):
        """Number of nonlinear elements presented in the system"""
        return len(self.circuit.nonlinear_elements.items())

    def get_epr_coefficients(self):
        """
        Return matrix of EPR coefficients with shape MxJ, where M is number of modes in the system and J is a number
        of nonlinear elements
        """
        epr_mat = np.empty((self.num_modes, self.num_nonlinear_elements))
        for j, nonlinear_element in enumerate(self.circuit.nonlinear_elements.values()):
            element_epr = self.circuit.element_epr([nonlinear_element])
            epr_mat[:, j] = np.real(element_epr) * 2
        return epr_mat

    def _get_flux_zpf(self):
        """Calculates the quantum zero point fluctuations matrix"""
        epr_mat = self.get_epr_coefficients()
        w = np.imag(self.circuit.w)
        w_mat = np.diag(w)
        ej_mat_inv = np.diag([1 / element.ej for element in self.circuit.nonlinear_elements.values()])
        phi_mj_2 = hbar / 2 * w_mat @ epr_mat @ ej_mat_inv
        return np.sqrt(phi_mj_2)

    def get_kerr_matrix(self):
        epr_mat = self.get_epr_coefficients()
        w = np.imag(self.circuit.w)
        w = w[w > 0]
        w_mat = np.diag(w)
        ej_mat_inv = np.diag([1 / element.ej for element in self.circuit.nonlinear_elements.values()])
        temp = w_mat @ epr_mat
        kerr_matrix = hbar / 4 * temp @ ej_mat_inv @ temp.T
        anharmonicities = np.diag(kerr_matrix) / 2
        return kerr_matrix, anharmonicities

    def get_hamiltonian_perturbation(self, modes: List[int], num_levels: List[int], order: int = 4,
                                     only_linear: bool = False):
        """
        Returns Hamiltonian function in harmonic mode basis up to defined order of non-linearity
        :param modes: list of mode indices
        :param num_levels: number of levels in each mode
        :param order: order of Lagrange expansion for non-linear elements
        :param only_linear: if True, when returns only linear part of Hamiltonian
        """
        assert len(modes) == len(num_levels)
        assert order > 3

        from itertools import product
        from math import factorial
        w = np.imag(self.circuit.w[modes])
        num_modes = len(w)
        ham = np.zeros((np.prod(num_levels), np.prod(num_levels)), dtype=complex)
        basis = np.asarray(list(product(*[tuple(np.arange(dim)) for dim in num_levels])))

        # linear part of hamiltonian
        ham += hbar * np.diag(basis @ w)
        phi_mj_zpf = self._get_flux_zpf()

        if not only_linear:
            for j, nonlinear_element in enumerate(self.circuit.nonlinear_elements.values()):
                li, *_ = nonlinear_element.get_lagrangian_series(order=order)
                phi_j_operator = np.zeros_like(ham)
                for mode_m in range(num_modes):
                    phi_j_mode_operator = np.identity(1)
                    for mode_n in range(num_modes):
                        if mode_n == mode_m:
                            operator = self.create(num_levels[mode_m]) + self.destroy(num_levels[mode_m])
                        else:
                            operator = np.identity(num_levels[mode_n])
                        phi_j_mode_operator = np.kron(phi_j_mode_operator, operator)

                    phi_j_operator += phi_mj_zpf[modes[mode_m], j] * phi_j_mode_operator

                for p in range(3, order + 1):
                    c_p = li[0, 0, p - 2] * PHI0 ** p / p
                    ham += c_p * np.linalg.matrix_power(phi_j_operator, p)
        return ham

    def get_kerr_matrix_hamiltonian(self, modes: List[int], num_levels: List[int], order: int = 4,
                                    threshold: float = 0.8):
        """
        Returns Kerr matrix from Hamiltonian up to defined order of non-linearity.
        The Kerr matrix is calculated using single photon and double photon excitation states in the dispersive limit.
        :param modes: list of mode indices
        :param num_levels: number of levels in each mode
        :param order: order of Lagrange expansion for non-linear elements
        :param threshold: threshold for ratios of dressed states
        """
        num_modes = len(modes)
        omegas = np.imag(self.circuit.w[modes])
        ham = self.get_hamiltonian_perturbation(modes, num_levels, order)
        energies, states = np.linalg.eigh(ham)

        # add single photon and double photon states
        undressed_states = []
        for mode_m in range(num_modes):
            for mode_n in range(num_modes):
                basis_state = [0] * num_modes
                basis_state[mode_m] = 1
                basis_state[mode_n] = 1
                undressed_states.append(tuple(basis_state))

        # add ground state
        undressed_states.append(tuple([0] * num_modes))
        dressed_state_ids, dressed_state_participation_ratios = self.find_dressed_state_id(states, num_levels,
                                                                                           undressed_states)

        dressed_state_ids = np.asarray(dressed_state_ids).ravel()
        dressed_state_participation_ratios = np.asarray(dressed_state_participation_ratios).ravel()

        if not all(dressed_state_participation_ratios > threshold):
            raise ValueError("Non-dispersive states were found in the list of dressed states")

        ground_state = states[:, dressed_state_ids[-1]].ravel()
        single_double_photon_states = states[:, dressed_state_ids[:-1]]

        ground = ground_state.T.conj() @ ham @ ground_state
        kerr = np.diag(single_double_photon_states.T.conj() @ ham @ single_double_photon_states)
        kerr = kerr.reshape((num_modes, num_modes))
        reference = np.zeros_like(kerr)
        np.fill_diagonal(reference, ground + hbar * omegas)

        for mode_m in range(num_modes):
            for mode_n in range(num_modes):
                if mode_m != mode_n:
                    reference[mode_m, mode_n] = kerr[mode_m, mode_m] + kerr[mode_n, mode_n] - ground
        energies = np.diag(kerr) - ground
        return energies, kerr - reference

    @staticmethod
    def fock(num: int, n: int):
        """
        Returns fock state for single mode
        :param num: number of excitations
        :param n:
        """
        assert num < n
        state = np.zeros(n, dtype=complex)
        state[num] = 1.
        return state

    @staticmethod
    def create(n: int):
        """Returns matrix representation of an n-dimensional creation operator"""
        diag = np.sqrt(np.arange(1, n))
        mat = np.zeros([n, n], dtype=complex)
        np.fill_diagonal(mat[1:], diag)
        return mat

    @staticmethod
    def destroy(n: int):
        """Returns matrix representation of an n-dimensional annihilation operator"""
        diag = np.sqrt(np.arange(1, n))
        mat = np.zeros([n, n], dtype=complex)
        np.fill_diagonal(mat[:, 1:], diag)
        return mat

    @staticmethod
    def kron_list(*args):
        """Kronecker product for list of operators"""
        if len(args) == 1:
            return args[0]
        else:
            prod = np.kron(args[0], args[1])
            for arg in args[2:]:
                prod = np.kron(prod, arg)
            return prod

    def find_dressed_state_id(self, states: np.ndarray, num_levels: List[int], undressed_states: List[Tuple[int, ...]],
                              num_states: int = 1):
        """
        Returns ids of dressed states closed to the defined undressed states and their participation ratios
        :param states: array of eigen vectors
        :param num_levels: number of levels in each mode
        :param undressed_states: defined undressed state
        :param num_states: number of states to return for maximum ratios
        """
        space_dim = np.prod(num_levels)
        basis_states = [[self.fock(state, num_levels[i]) for i, state in enumerate(basis_state)] for
                        basis_state in undressed_states]

        undressed_states = np.zeros((space_dim, len(basis_states)), dtype=complex)
        for basis_state_id, basis_state in enumerate(basis_states):
            undressed_states[:, basis_state_id] = self.kron_list(*basis_state)
        participation_ratios = np.abs(states.T @ undressed_states) ** 2
        sorting = np.argsort(participation_ratios, axis=0)
        sorting = np.flip(sorting, axis=0)[:num_states]

        dressed_state_ids = [list(sorting[:, state_id]) for state_id in range(len(basis_states))]
        dressed_state_participation_ratios = [list(participation_ratios[closest_state_ids, state_id]) for
                                              state_id, closest_state_ids in enumerate(dressed_state_ids)]
        return dressed_state_ids, dressed_state_participation_ratios
