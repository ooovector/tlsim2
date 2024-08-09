import unittest
import sys
import numpy as np
from scipy.constants import h, hbar, e
from tlsim2.lumped import LumpedTwoTerminal
from tlsim2.lumped import JosephsonJunction
from tlsim2.circuit import Circuit
from tlsim2.quantum_circuit import QuantumCircuit

class QuantumCircuitTests(unittest.TestCase):
    def test_quantum_circuit(self):
        Cq = 73e-15
        EJ = 16e9 * h

        # qubit capacitor
        capacitor_qubit = LumpedTwoTerminal(name='Cq', c=Cq)

        # qubit nonlinear element
        jj = JosephsonJunction(name='JJ', ej=EJ)

        linear_circuit = Circuit()
        linear_circuit.add_element(capacitor_qubit, {'i': 0, 'o': 1})
        linear_circuit.add_element(jj, {'i': 0, 'o': 1})
        linear_circuit.short(0)

        w, v, node_names = linear_circuit.compute_system_modes()
        decays = np.real(w/1e9)*2
        frequencies = np.imag(w/(2*np.pi))/1e9
        decays = np.real(w/1e9)*2
        decays = decays[frequencies > 0]
        frequencies = frequencies[frequencies > 0]
        assert np.allclose(frequencies, [5.8278845])
        assert np.allclose(decays, [0])

        nonlinear_circuit = QuantumCircuit(linear_circuit)

        kerr_matrix, anharmonicities = nonlinear_circuit.get_kerr_matrix()

        f_p = frequencies[0]
        alpha = - anharmonicities[0] / (2 * np.pi) / 1e6

        print(f"Plasma frequency for transmon qubit is {f_p :.2f} GHz and \
        anharmonicity {alpha :.2f} MHz, therefore qubit transition is {f_p + alpha / 1e3 :.2f} GHz")


if __name__ == '__main__':
    unittest.main()
