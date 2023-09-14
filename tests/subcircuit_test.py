import unittest
import numpy as np
from tlsim2.tl import default_tl_basis, MultiTransmissionLine
from tlsim2.lumped import LumpedTwoTerminal
from tlsim2.circuit import Circuit


class SubcircuitTest(unittest.TestCase):
    def test_subcircuit(self, plots=False):
        c1 = LumpedTwoTerminal('C', l=np.inf, c=1, r=np.inf)
        tl1 = MultiTransmissionLine('TL1', n=1, l=1, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 0))

        c2 = LumpedTwoTerminal('C2', l=np.inf, c=1, r=np.inf)
        tl2 = MultiTransmissionLine('TL2', n=1, l=1, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 0))

        res1 = Circuit(name='res1')
        res1.add_element(tl1, {'i0': 0, 'o0': 1})
        res1.add_element(c1, {'i': 2, 'o': 1})
        res1.short(0)
        print('Small circuit node_names: ', res1.node_names)

        w, v, node_names = res1.compute_system_modes()
        print('Small circuit w: ', w)
        # print('Small circuit v: ', v[:4, [0, 4, 5, 7]])
        print('Small circuit v: ', v.T)
        res1_el = res1.make_element(nodes={2: 'res1'}, cutoff_low=0, cutoff_high=3)

        # li, c, ri = res1_el.get_element_licri()
        # print (res1_el.get_terminal_names())
        li, c, ri, node_names = res1.get_system_licri()
        print('Small circuit li: \n', np.round(li, 3))
        print('Small circuit c: \n', np.round(c, 3))
        print('Small circuit ri: \n', np.round(ri, 3))
        print('Small circuit node_names: ', node_names)

        li, c, ri = res1_el.get_element_licri()
        print('Small element li: \n', np.round(li, 3))
        print('Small element c: \n', np.round(c, 3))
        print('Small element:ri \n', np.round(ri, 3))
        print('Small element node_names: ', res1_el.get_terminal_names())

        #
        res2 = Circuit(name='res2')
        res2.add_element(tl2, {'i0': 0, 'o0': 1})
        res2.add_element(c2, {'i': 2, 'o': 1})
        res2.short(0)
        w, v, node_names = res2.compute_system_modes()
        # # print(w)
        res2_el = res2.make_element(nodes={2: 'res2'}, cutoff_low=0, cutoff_high=3)
        # #
        sample = Circuit()
        sample.add_element(res1_el, {'res1': 2})
        sample.add_element(res2_el, {'res2': 2})
        # # #
        li, c, ri, node_names = sample.get_system_licri()
        print('li system: \n', np.round(li, 3))
        print('c system: \n', np.round(c, 3))
        print('ri system: \n', np.round(ri, 3))
        w, v, node_names = sample.compute_system_modes()
        #
        # print (w, v)
        print('Big circuit w: ', w)
        print('Big circuit node_name: ', node_names)

        # assert np.sum(np.abs(np.abs(np.imag(w)) - 0.86033) < 1e-5) == 2
        print('Found shunted quarter wavelength resonators!')


        print('Big circuit')
        sample = Circuit()
        sample.add_element(tl1, {'i0': 0, 'o0': 1})
        sample.add_element(c1, {'i': 2, 'o': 1})
        sample.add_element(tl2, {'i0': 0, 'o0': 3})
        sample.add_element(c2, {'i': 2, 'o': 3})
        sample.short(0)

        li, c, ri, node_names = sample.get_system_licri()
        print('li system: \n', np.round(li, 3))
        print('c system: \n', np.round(c, 3))
        print('ri system: \n', np.round(ri, 3))
        w, v, node_names = sample.compute_system_modes()
        print ('w:', w)

        return w


if __name__ == '__main__':
    unittest.main()
