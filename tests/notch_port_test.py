import unittest
import numpy as np
from tlsim2.tl import default_tl_basis, MultiTransmissionLine
from tlsim2.lumped import LumpedTwoTerminal
from tlsim2.circuit import Circuit
from matplotlib import pyplot as plt


class NotchPortTest(unittest.TestCase):
    def test_resonator(self):
        # defining a resontor coupled to a matched TL

        cl_coupler = np.asarray([[167.229, -14.341], [-14.341, 167.229]])*1e-12
        ll_coupler = np.asarray([[417.246, 35.781], [35.781, 417.246]])*1e-9

        cl_res = [[413.098e-12]]
        ll_res = [[167.666e-9]]

        length = 5.0e-3

        tl = MultiTransmissionLine('coupler', n=2, l=length, ll=ll_res,
                                        cl=cl_res, basis=default_tl_basis(1, 1))

        sys = Circuit('ResonatorTest')
        sys.add_element(tl, {'i0': 'tl_open', 'o0': 'tl_short'})

        sys.short('tl_short')

        li, c, ri, node_names = sys.get_system_licri()
        frequencies, modes, node_names = sys.compute_system_modes()
        print(frequencies/(2*np.pi))

    def test_notch_port(self):
        # defining a resontor coupled to a matched TL
        p1 = LumpedTwoTerminal('p1', r=50)
        p2 = LumpedTwoTerminal('p2', r=50)

        cl_coupler = np.asarray([[167.229, -14.341], [-14.341, 167.229]])*1e-12
        ll_coupler = np.asarray([[417.246, 35.781], [35.781, 417.246]])*1e-9

        cl_res = [[167.666e-12]]
        ll_res = [[413.098e-9]]

        coupler_length = 0.4e-3
        shorted_length = 3.6e-3
        open_length  = 1.0e-3

        q = 1/((np.sin(np.pi/2*coupler_length/5e-3)*14.341/167.229)**2/(np.pi/2))

        print(q)

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
        # epr_res = sys.element_epr([shorted_end], modes)
        epr_res_modes = np.argsort(-epr_res)
        print(frequencies[epr_res_modes[:8]]/(2*np.pi))
        print(np.imag(frequencies[epr_res_modes[:8]])/(2*np.pi),
              np.imag(frequencies[epr_res_modes[:8]])/(2*np.real(frequencies[epr_res_modes[:8]])) )
        # print(frequencies/(2*np.pi))
        # li, c, ri = coupler.get_element_licri()
        print('notch port coupled li:', np.real(li))
        print('notch port coupled c:', np.real(c))
        # print('terminals:', coupler.get_terminal_names())
        # #
        # li, c, ri = open_end.get_element_licri()
        # print('li:', np.round(np.real(li), -7))
        # print('c:', np.round(np.real(c), 16))
        # print('terminals:', open_end.get_terminal_names())

        # li, c, ri, node_names = sys.get_system_licri()
        # # plt.pcolormesh(np.real(c), cmap='nipy_spectral')
        # plt.pcolormesh(np.linalg.pinv(np.real(li)), cmap='nipy_spectral')
        # print('li:', np.real(li))
        # print('c:', np.real(c))
        # plt.xticks(np.arange(c.shape[0]))
        # plt.gca().set_xticklabels(node_names, rotation=70)
        # plt.yticks(np.arange(c.shape[0]))
        # plt.gca().set_yticklabels(node_names)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

    def test_notch_port_capacitive(self):
        # defining a resontor coupled to a matched TL
        p1 = LumpedTwoTerminal('p1', r=50)
        # p2 = LumpedTwoTerminal('p2', r=50)

        cl_coupler = np.asarray([[167.229, -14.341], [-14.341, 167.229]])*1e-12
        ll_coupler = np.asarray([[417.246, 35.781], [35.781, 417.246]])*1e-9

        cl_res = [[167.666e-12]]
        ll_res = [[413.098e-9]]

        coupler_length = 0.4e-3
        open_length = 0.1e-3
        # shorted_length = 4.5e-3
        shorted_length = 5.0e-3

        q = 1/((np.sin(np.pi/2*coupler_length/5e-3)*14.341/167.229)**2/(np.pi/2))

        print(q)

        coupler = LumpedTwoTerminal('C', c=-cl_coupler[0, 1]*coupler_length)
        # open_end = MultiTransmissionLine('open_end', n=1, l=open_length, ll=ll_res,
        #                                  cl=cl_res, basis=default_tl_basis(1, 1))
        shorted_end = MultiTransmissionLine('shorted_end', n=1, l=shorted_length, ll=ll_res,
                                            cl=cl_res, basis=default_tl_basis(1, 0))

        sys = Circuit('NotchPortTest')
        sys.add_element(p1, {'i': 'p1', 'o': 'gnd'})
        # sys.add_element(p2, {'i': 'p1', 'o': 'gnd'})
        sys.add_element(coupler, {'i': 'p1', 'o': 'open_start'})
        # sys.add_element(open_end, {'i0': 'open_start', 'o0': 'open_end'})
        sys.add_element(shorted_end, {'i0': 'open_start', 'o0': 'gnd'})

        sys.short('gnd')

        li, c, ri, node_names = sys.get_system_licri()
        frequencies, modes, node_names = sys.compute_system_modes()

        # epr_res = sys.element_epr([open_end, shorted_end], modes)
        epr_res = sys.element_epr([shorted_end], modes)
        epr_res_modes = np.argsort(-epr_res)
        print('Capacitively coupled frequencies: ', frequencies[epr_res_modes[:12]]/(2*np.pi))
        print('Capacitively coupled decay/quality factors: ', np.imag(frequencies[epr_res_modes[:12]])/(2*np.pi),
              np.imag(frequencies[epr_res_modes[:12]])/(2*np.real(frequencies[epr_res_modes[:12]])) )
        # print(frequencies/(2*np.pi))
        # li, c, ri = coupler.get_element_licri()
        print('system li:', np.real(li))
        print('system c:', np.real(c))
        li, c, ri = shorted_end.get_element_licri()
        print('tl li:', np.real(li))
        print('tl c:', np.real(c))
        # print('terminals:', coupler.get_terminal_names())
        # #
        # li, c, ri = open_end.get_element_licri()
        # print('li:', np.round(np.real(li), -7))
        # print('c:', np.round(np.real(c), 16))
        # print('terminals:', open_end.get_terminal_names())

        # li, c, ri, node_names = sys.get_system_licri()
        # # plt.pcolormesh(np.real(c), cmap='nipy_spectral')
        # plt.pcolormesh(np.linalg.pinv(np.real(li)), cmap='nipy_spectral')
        # print('li:', np.real(li))
        # print('c:', np.real(c))
        # plt.xticks(np.arange(c.shape[0]))
        # plt.gca().set_xticklabels(node_names, rotation=70)
        # plt.yticks(np.arange(c.shape[0]))
        # plt.gca().set_yticklabels(node_names)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

if __name__ == '__main__':
    unittest.main()
