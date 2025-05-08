import unittest
import numpy as np
from tlsim2.tl import default_tl_basis, MultiTransmissionLine, MultiTransmissionLineFromGeometry
from tlsim2.lumped import LumpedTwoTerminal
from tlsim2.circuit import Circuit
from matplotlib import pyplot as plt


class TestComponents(unittest.TestCase):

    def setUp(self):
        self.tl_coupler = MultiTransmissionLineFromGeometry(name='tl_coupler', widths=[7., 7.], gaps=[4., 4., 4.],
                                                            length=300.)


    def test_components(self):

        # capacitance matrix per pF/m
        cl = np.asarray([[165.676,	-58.753],
                         [-58.753,	165.676]])

        # inductance matrix per nH/m
        ll = np.asarray([[478.197, 	169.581],
                         [169.581,	478.197]])

        np.testing.assert_allclose(cl, self.tl_coupler.cl * 1e12 / 1e-6, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(ll, self.tl_coupler.ll * 1e9 / 1e-6, rtol=1e-4, atol=1e-5)


class CircuitTest(unittest.TestCase):
    def test_resonator_circuit(self, plots=False):

        l = LumpedTwoTerminal('L', l=1, c=0, r=np.inf)
        c = LumpedTwoTerminal('C', l=np.inf, c=1, r=np.inf)
        r = LumpedTwoTerminal('R', l=np.inf, c=0, r=100)
        # tl = MultiTransmissionLine('TL1', n=1, l=2, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))

        sys = Circuit()
        sys.add_element(l, {'i': 0, 'o': 1})
        sys.add_element(c, {'i': 0, 'o': 1})
        # sys.add_element(tl, {'i0': 0, 'o0': 1})
        sys.add_element(r, {'i': 0, 'o': 1})

        sys.short(0)

        li_sys, c_sys, ri_sys, nodes = sys.get_system_licri()
        # print('licri: ', li_sys, c_sys, ri_sys, nodes)
        w, v, modes = sys.compute_system_modes()

        print('Parallel LRC angular frequencies:', w)

        if plots:
            xlist = np.linspace(-0.5, 0.5, 21)
            print('modes: ', w, v, modes)
            for mode_id in range(len(w)):
                fd = np.zeros(len(xlist), complex)
                for basis_element_id, basis_element in enumerate(tl.basis.elements):
                    # print(len(basis_element.elements), basis_element.elements)
                    for x_id, x in enumerate(xlist):
                        val = (v[basis_element_id, mode_id] * basis_element.elements[0]).eval(x)
                        fd[x_id] += val
                        print (f'Calculating product {basis_element_id}, {mode_id}, {x_id}: {v[basis_element_id, mode_id]}, {basis_element.elements[0].eval(x)}')
                plt.plot(np.real(fd), label='real '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))
                plt.plot(np.imag(fd), label='imag '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))

            plt.legend()
            plt.show()
            # return w

    def test_resonator_circuit(self, plots=False):

        l = LumpedTwoTerminal('L', l=1, c=0, r=np.inf)
        c = LumpedTwoTerminal('C', l=np.inf, c=1, r=np.inf)
        r = LumpedTwoTerminal('R', l=np.inf, c=0, r=100)
        # tl = MultiTransmissionLine('TL1', n=1, l=2, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))

        sys = Circuit()
        sys.add_element(l, {'i': 0, 'o': 1})
        sys.add_element(c, {'i': 0, 'o': 1})
        # sys.add_element(tl, {'i0': 0, 'o0': 1})
        sys.add_element(r, {'i': 0, 'o': 1})

        sys.short(0)

        li_sys, c_sys, ri_sys, nodes = sys.get_system_licri()
        # print('licri: ', li_sys, c_sys, ri_sys, nodes)
        w, v, modes = sys.compute_system_modes()

        print('Parallel LRC angular frequencies:', w)

        if plots:
            xlist = np.linspace(-0.5, 0.5, 21)
            print('modes: ', w, v, modes)
            for mode_id in range(len(w)):
                fd = np.zeros(len(xlist), complex)
                for basis_element_id, basis_element in enumerate(tl.basis.elements):
                    # print(len(basis_element.elements), basis_element.elements)
                    for x_id, x in enumerate(xlist):
                        val = (v[basis_element_id, mode_id] * basis_element.elements[0]).eval(x)
                        fd[x_id] += val
                        print (f'Calculating product {basis_element_id}, {mode_id}, {x_id}: {v[basis_element_id, mode_id]}, {basis_element.elements[0].eval(x)}')
                plt.plot(np.real(fd), label='real '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))
                plt.plot(np.imag(fd), label='imag '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))

            plt.legend()
            plt.show()
            # return w

    def test_tl(self):
        tl = MultiTransmissionLine('TL1', n=1, l=2, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))
        print('TL licri:', *tl.get_element_licri())

    def test_two_section_tl(self, plots=False):
        tl1 = MultiTransmissionLine('TL1', n=1, l=0.5, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))
        tl2 = MultiTransmissionLine('TL2', n=1, l=1.5, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))

        sys = Circuit()
        sys.add_element(tl1, {'i0': 0, 'o0': 1})
        sys.add_element(tl2, {'i0': 1, 'o0': 2})

        sys.short(0)

        w, v, node_names = sys.compute_system_modes()
        element_modes = sys.system_modes_to_element_modes(v, [tl1])

        if plots:
            xlist = np.linspace(-0.5, 0.5, 21)
            for mode_id in range(len(w)):
                fd = np.zeros(len(xlist), complex)
                for element_mode, basis_element in zip(element_modes[mode_id][0].values(), tl1.basis.elements):
                    # print(len(basis_element.elements), basis_element.elements)
                    for x_id, x in enumerate(xlist):
                        val = (element_mode * basis_element.elements[0]).eval(x)
                        fd[x_id] += val
                        # print (f'Calculating product2 (), {mode_id}, {x_id}: {element_mode}, {basis_element.elements[0].eval(x)}')
                plt.plot(np.real(fd), label='real '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))
                plt.plot(np.imag(fd), label='imag '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))

                print('system licri:', sys.get_system_licri())

            plt.legend()
            plt.show()

        # We should get two quarter-wavelength modes
        assert np.sum(np.abs(np.abs(np.imag(w)) - np.pi/4) < 5e-5) == 2
        print('Found quarter wavelength resonators with two sections!')

        return w

    def test_quarter_wavelength_resonator(self, plots=False):
        tl = MultiTransmissionLine('TL1', n=1, l=1, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))

        sys = Circuit()
        sys.add_element(tl, {'i0': 0, 'o0': 1})

        sys.short(0)

        w, v, node_names = sys.compute_system_modes()
        if plots:
            xlist = np.linspace(-0.5, 0.5, 21)

        element_modes = sys.system_modes_to_element_modes(v, [tl])

        if plots:
            for mode_id in range(len(w)):
                fd = np.zeros(len(xlist), complex)
                for element_mode, basis_element in zip(element_modes[mode_id][0].values(), tl.basis.elements):
                    # print(len(basis_element.elements), basis_element.elements)
                    for x_id, x in enumerate(xlist):
                        val = (element_mode * basis_element.elements[0]).eval(x)
                        fd[x_id] += val
                        # print (f'Calculating product2 (), {mode_id}, {x_id}: {element_mode}, {basis_element.elements[0].eval(x)}')
                plt.plot(np.real(fd), label='real '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))
                plt.plot(np.imag(fd), label='imag '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))

                print('system licri:', sys.get_system_licri())

            plt.legend()
            plt.show()

        # We should get two quarter-wavelength modes
        assert np.sum(np.abs(np.abs(np.imag(w)) - np.pi/2) < 5e-5) == 2
        print('Found quarter wavelength resonators!')

        return w

    def test_shunted_quarter_wavelength_resonator(self, plots=False):
        c = LumpedTwoTerminal('C', l=np.inf, c=1, r=np.inf)
        tl = MultiTransmissionLine('TL1', n=1, l=1, cl=[[1]], ll=[[1]], basis=default_tl_basis(1, 1))

        sys = Circuit()
        sys.add_element(c, {'i': 0, 'o': 1})
        sys.add_element(tl, {'i0': 0, 'o0': 1})

        sys.short(0)

        w, v, node_names = sys.compute_system_modes()
        element_modes = sys.system_modes_to_element_modes(v, [tl])
        if plots:
            xlist = np.linspace(-0.5, 0.5, 21)
            for mode_id in range(len(w)):
                fd = np.zeros(len(xlist), complex)
                print(element_modes[mode_id], tl.basis.elements)
                for element_mode, basis_element in zip(element_modes[mode_id][0].values(), tl.basis.elements):
                    # print(len(basis_element.elements), basis_element.elements)
                    for x_id, x in enumerate(xlist):
                        val = (element_mode * basis_element.elements[0]).eval(x)
                        fd[x_id] += val
                        # print (f'Calculating product2 (), {mode_id}, {x_id}: {element_mode}, {basis_element.elements[0].eval(x)}')
                plt.plot(np.real(fd), label='real '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))
                plt.plot(np.imag(fd), label='imag '+str(mode_id)+' E: '+str(np.round(w[mode_id], 5)))

                print('system licri:', sys.get_system_licri())

            plt.legend()
            plt.show()

        assert np.sum(np.abs(np.abs(np.imag(w)) - 0.86033) < 1e-5) == 2
        print('Found shunted quarter wavelength resonators!')

        return w

    def test_quality_factor_gopplpaper(self):
        samples = {'A': {'C': 56.4e-15, 'f0': 2.2678e9, 'l': 28.449e-3,
                         'RL': 50, 'QL': 3.7e2, 'Z0':59.7, 'Cl': 1.27e-10},
                   'B': {'C': 48.6e-15, 'f0': 2.2763e9, 'l': 28.449e-3,
                         'RL': 50, 'QL': 4.9e2, 'Z0': 59.7, 'Cl': 1.27e-10},
                   'C': {'C': 42.9e-15, 'f0': 2.2848e9, 'l': 28.449e-3,
                         'RL': 50, 'QL': 7.5e2, 'Z0': 59.7, 'Cl': 1.27e-10},
                   'D': {'C': 35.4e-15, 'f0': 2.2943e9, 'l': 28.449e-3,
                         'RL': 50, 'QL': 1.1e3, 'Z0': 59.7, 'Cl': 1.27e-10},
                   'E': {'C': 26.4e-15, 'f0': 2.3086e9, 'l': 28.449e-3,
                         'RL': 50, 'QL': 1.7e3, 'Z0': 59.7, 'Cl': 1.27e-10},
                   }

        for sample_name, sample in samples.items():
            c1 = LumpedTwoTerminal('C1', l=np.inf, c=sample['C'], r=np.inf)
            rl1 = LumpedTwoTerminal('RL1', l=np.inf, c=0, r=sample['RL'])
            c2 = LumpedTwoTerminal('C2', l=np.inf, c=sample['C'], r=np.inf)
            rl2 = LumpedTwoTerminal('RL2', l=np.inf, c=0, r=sample['RL'])
            cl = sample['Cl'] # per-unit-length capacitance
            ll = sample['Z0']**2*cl# per-unit-length inductance
            tl = MultiTransmissionLine('TL1', n=1, l=sample['l'], cl=[[cl]], ll=[[ll]], basis=default_tl_basis(1, 1))

            sys = Circuit()
            sys.add_element(rl1, {'i': 0, 'o': 1})
            sys.add_element(c1, {'i': 1, 'o': 2})
            sys.add_element(tl, {'i0': 2, 'o0': 3})
            sys.add_element(c2, {'i': 3, 'o': 4})
            sys.add_element(rl2, {'i': 4, 'o': 0})

            sys.short(0)

            w, v, node_names = sys.compute_system_modes()

            # print(w, v)

            # print('Frequencies [GHz]: ', np.round(np.imag(w)/(2*np.pi*1e9), 4))
            # print('Quality factors: ', np.round(np.imag(w)/np.real(w)/2, 3))

            closest_mode_id = np.argmin(np.abs(np.abs(np.imag(w)/(2*np.pi))-sample['f0']))
            print(f'Sample {sample_name}: {np.round(np.abs(np.imag(w)/(2*np.pi*1e9))[closest_mode_id], 4)} {np.round(sample["f0"]/1e9, 4)}')


if __name__ == '__main__':
    unittest.main()
