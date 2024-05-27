from typing import List, Any
import numpy as np
from .field_distribution import FieldDistribution


class LinearHull:
    elements: List[FieldDistribution]
    basis_products: np.array

    def __init__(self, elements: List[FieldDistribution], real=True):
        """
        Linear Hull: subspace spanned by a finite set of a number of vectors.
        Vectors should implement __add__(), __sub__(), __matmul__() on each other, and __mul__() with complex.
        :param elements:
        :param enforce real elements:
        """
        self.elements = elements
        self.real = real

        basis_products = np.zeros((len(self.elements),
                                   len(self.elements[0]),
                                   len(self.elements),
                                   len(self.elements[0])), dtype=complex)
        for i, basis_element1 in enumerate(self.elements):
            for k in range(len(self.elements[0])):
                for j, basis_element2 in enumerate(self.elements):
                    for l in range(len(self.elements[0])):
                        try:
                            basis_products[i, k, j, l] = basis_element1[k].conjugate() @ basis_element2[l]
                        except TypeError:
                            basis_products[i, k, j, l] = basis_element1[k].conjugate() * basis_element2[l]
        self.basis_products = basis_products

        if hasattr(self.elements[0], 'derivative'):
            basis_derivative_products = np.zeros((len(self.elements),
                                                  len(self.elements[0]),
                                                  len(self.elements),
                                                  len(self.elements[0])), dtype=complex)
            for i, basis_element1 in enumerate(self.elements):
                for k in range(len(self.elements[0])):
                    for j, basis_element2 in enumerate(self.elements):
                        for l in range(len(self.elements[0])):
                            try:
                                basis_derivative_products[i, k, j, l] = basis_element1[k].derivative().conjugate() @ \
                                                                        basis_element2[l].derivative()
                            except TypeError:
                                basis_derivative_products[i, k, j, l] = basis_element1[k].derivative().conjugate() * \
                                                                        basis_element2[l].derivative()

            self.basis_derivative_products = basis_derivative_products

        if self.real:
            self.basis_products = np.real(self.basis_products)
            if hasattr(self.elements[0], 'derivative'):
                self.basis_derivative_products = np.real(self.basis_derivative_products)

    def __repr__(self):
        return f'LinearHull with {len(self.elements)} elements:\n{self.elements}'

    def orthogonalize(self, epsilon=1e-11):
        eigenvalues, eigenvectors = np.linalg.eigh(self.basis_products.sum(axis=(1, 3)))
        print('printing ev, ev: ', eigenvalues, eigenvectors)
        # eigenvectors = eigenvectors.T
        new_basis = []

        for element_id, eigenvalue in enumerate(eigenvalues):
            if eigenvalue > epsilon:
                new_element = 0

                global_phase = np.angle(eigenvectors[np.argmax(np.abs(eigenvectors[:, element_id])), element_id])
                phase_multiplier = np.exp(-1j*global_phase)

                for coef, old_element in zip(eigenvectors[:, element_id], self.elements):
                    new_element += old_element * (phase_multiplier * coef)

                norm = np.abs(np.sqrt(new_element.conjugate() @ new_element))
                new_element = new_element * (1 / norm)

                new_basis.append(new_element)

        return LinearHull(new_basis)

    def gram_schmidt(self, epsilon=1e-9):
        """
        Perform Gram-Schmidt orthogonalization process
        :param epsilon:
        :return: LinearHull spanning the same subspace, but with orthogonal elements.
        """
        new_basis = []

        for element in self.elements:
            norm = np.sqrt(element.conjugate() @ element)
            normalized_element = element * (1 / norm)
            for previous_element in new_basis:
                normalized_element -= (previous_element.conjugate() @ normalized_element) * previous_element
            norm = np.sqrt(normalized_element.conjugate() @ normalized_element)
            if norm < epsilon:  # nothing left from the element:
                continue
            new_basis.append(normalized_element * (1 / norm))

        transform = np.zeros((len(new_basis), len(self.elements)), dtype=complex)

        for old_element_id, old_element in enumerate(self.elements):
            for new_element_id, new_element in enumerate(new_basis):
                transform[new_element_id, old_element_id] = new_element.conjugate() @ old_element

        return LinearHull(new_basis), transform

    def gram_schmidt_terminals(self, terminals, epsilon=1e-9):
        """

        :param epsilon:
        :return:
        """
        element_fields_on_terminals = []

        for element in self.elements:
            port_fields = element.eval_terminals()
            element_fields_on_terminals.append([port_fields[terminal] for terminal in terminals])

        transform_terminals = np.linalg.pinv(element_fields_on_terminals)

        # the rest of the elements must be:
        # - have zeros in the terminals (which means we need to subtract the new_basis)

        basis_terminals = []
        for terminal_id in range(transform_terminals.shape[0]):
            new_basis_element = 0
            for element_id, element in enumerate(self.elements):
                # print (element, transform_terminals[terminal_id, element_id])
                new_basis_element += transform_terminals[terminal_id, element_id]*element
            basis_terminals.append(new_basis_element)
        #
        # print('Length of basis_terminals: ', len(basis_terminals))
        # basis_terminals_ortho = LinearHull(basis_terminals).orthogonalize().elements
        # print(f'New basis ortho (len {len(basis_terminals)}): ')

        internal_modes = []

        for element_id, element in enumerate(self.elements):
            internal_element = element.copy()
            for terminal_mode_id, terminal_mode in enumerate(basis_terminals):
                internal_element -= element_fields_on_terminals[element_id][terminal_mode_id] * terminal_mode

            internal_modes.append(internal_element)
            if len(LinearHull(basis_terminals + internal_modes).orthogonalize().elements) < \
                    (len(basis_terminals) + len(internal_modes)):
                del internal_modes[-1]
            #
            # print ('Adding element ', element_id, ' norm ', norm)

        print('len(basis_terminals): ', len(basis_terminals))
        print('len(internal_modes): ', len(internal_modes))

        return LinearHull(basis_terminals + internal_modes)

