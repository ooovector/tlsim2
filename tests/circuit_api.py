import unittest
import numpy as np
from tlsim2.tl import default_tl_basis, MultiTransmissionLine
from tlsim2.lumped import LumpedTwoTerminal
from tlsim2.circuit import Circuit


class ApiTest(unittest.TestCase):
    def test_circuit(self):
        # defining a resontor coupled to a matched TL
        p1 = LumpedTwoTerminal('p1', r=50)
        p2 = LumpedTwoTerminal('p2', r=50)

        cl_coupler = np.asarray([[167.229, -14.341], [-14.341, 167.229]])*1e-12
        ll_coupler = np.asarray([[417.246, 35.781], [35.781, 417.246]])*1e-9

        cl_res = [[167.666e-12]]
        ll_res = [[413.098e-9]]

        coupler_length = 0.4e-3
        shorted_length = 3.6e-3
        open_length = 1.0e-3


        coupler = MultiTransmissionLine('coupler', n=2, l=coupler_length, ll=ll_coupler,
                                        cl=cl_coupler, basis=default_tl_basis(2, 0))
        open_end = MultiTransmissionLine('open_end', n=1, l=open_length, ll=ll_res,
                                         cl=cl_res, basis=default_tl_basis(1, 0))
        shorted_end = MultiTransmissionLine('shorted_end', n=1, l=shorted_length, ll=ll_res,
                                            cl=cl_res, basis=default_tl_basis(1, 1))

        sys = Circuit('NotchPortTest')
        sys.add_element(p1, {'i': 'p1', 'o': 'gnd'})
        sys.add_element(p2, {'i': 'p2', 'o': 'gnd'})
        sys.add_element(coupler, {'i0': 'p1', 'o0': 'p2', 'i1': 'open_start', 'o1': 'shorted_start'})
        sys.add_element(open_end, {'i0': 'open_start', 'o0': 'open_end'})
        sys.add_element(shorted_end, {'i0': 'shorted_start', 'o0': 'gnd'})

        sys.short('gnd')

        li, c, ri, node_names = sys.get_system_licri()
        frequencies, modes, node_names = sys.compute_system_modes()

        epr_res = sys.element_epr([open_end, shorted_end], modes)
        epr_res_modes = np.argsort(-epr_res)


if __name__ == '__main__':
    unittest.main()
