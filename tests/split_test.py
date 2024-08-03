import unittest
from tlsim2.tl import default_tl_basis, MultiTransmissionLine
from tlsim2.circuit import Circuit
from tlsim2.lumped import LumpedTwoTerminal
import numpy as np


class SplitTest(unittest.TestCase):
    def test_autosplit(self):
        sys_rpf = Circuit('sys_rpf')

        ll_coupler = [[476.595, 169.262],
                      [169.262, 476.595]]  # per-unit-length inductance for two close 50-ohm TLs on cold silicon
        cl_coupler = [[166.304, -59.062],
                      [-59.062, 166.304]]  # per-unit-length capacitance for two close 50-ohm TLs on cold silicon
        ll = [[416.120e-9]]  # per-unit-length inductance for a 50-ohm TL on cold silicon
        cl = [[166.448e-12]]  # per-unit-length capacitance for a 50-ohm TL on cold silicon

        t = 2
        t2 = 1

        rp_length = 0.0002
        pf_length = 0.0004
        rs_length = 0.001
        ro_length = 0.003
        ps_length = 0.001
        pm_length = 0.0015
        po_length = 0.002

        r_cap = 10e-15
        f_res = 50

        default_basis2 = default_tl_basis(2, t2)
        default_basis = default_tl_basis(1, t)

        rp = MultiTransmissionLine('rp', n=2, l=rp_length, ll=np.asarray(ll_coupler) * 1e-9,
                                   cl=np.asarray(cl_coupler) * 1e-12, basis=default_basis2, coupler_hint=True)
        # rp = MultiTransmissionLine('rp', n=1, l=rp_length, ll=ll, cl=cl, basis=default_basis)

        pf = MultiTransmissionLine('pf', n=2, l=pf_length, ll=np.asarray(ll_coupler) * 1e-9,
                                   cl=np.asarray(cl_coupler) * 1e-12, basis=default_basis2, coupler_hint=True)

        rs = MultiTransmissionLine('rs', n=1, l=rs_length, ll=ll, cl=cl, basis=default_basis)
        ro = MultiTransmissionLine('ro', n=1, l=ro_length, ll=ll, cl=cl, basis=default_basis)
        ps = MultiTransmissionLine('ps', n=1, l=ps_length, ll=ll, cl=cl, basis=default_basis)
        pm = MultiTransmissionLine('pm', n=1, l=pm_length, ll=ll, cl=cl, basis=default_basis)
        po = MultiTransmissionLine('po', n=1, l=ps_length, ll=ll, cl=cl, basis=default_basis)

        cap = LumpedTwoTerminal(name='c', c=r_cap, keep_top_level=True)
        # cap = LumpedTwoTerminal(name='c', c=r_cap, keep_top_level=False)
        p1 = LumpedTwoTerminal(name='p1', r=f_res)
        p2 = LumpedTwoTerminal(name='p2', r=f_res)

        sys_rpf.add_element(pf, {'i0': 1, 'o0': 2, 'i1': 3, 'o1': 4})
        sys_rpf.add_element(p1, {'i': 1, 'o': 0})
        sys_rpf.add_element(p2, {'i': 2, 'o': 0})  # notch-port-coupled resonator

        sys_rpf.add_element(ps, {'i0': 3, 'o0': 0})
        sys_rpf.add_element(pm, {'i0': 4, 'o0': 5})
        sys_rpf.add_element(rp, {'i0': 5, 'o0': 6, 'i1': 7, 'o1': 8})
        # sys_rpf.add_element(rp, {'i0': 5, 'o0': 6})
        sys_rpf.add_element(po, {'i0': 6, 'o0': 9})

        sys_rpf.add_element(ro, {'i0': 7, 'o0': 10})
        sys_rpf.add_element(rs, {'i0': 8, 'o0': 0})

        sys_rpf.add_element(cap, {'i': 10, 'o': 0})

        sys_rpf.short(0)

        w_rpf, v_rpf, node_names_rpf = sys_rpf.compute_system_modes()

        frequencies_rpf = np.imag(w_rpf / (2 * np.pi)) / 1e9
        decays_rpf = np.real(w_rpf / 1e9) * 2
        mask_rpf = frequencies_rpf > 0
        decays_rpf = decays_rpf[mask_rpf]
        frequencies_rpf = frequencies_rpf[mask_rpf]

        split_system = sys_rpf.autosplit()
        resonator = split_system.find_subsystem_name_by_element('rs')
        split_system.rename_element(resonator, 'resonator')
        purcell = split_system.find_subsystem_name_by_element('ps')
        split_system.rename_element(purcell, 'Purcell-filter')
        feedline = split_system.find_subsystem_name_by_element('p1')
        split_system.rename_element(feedline, 'feedline')

        ## find frequency close to the resonator and to the purcell
        w_split, v_split, node_names_rpf = split_system.compute_system_modes()
        resonator_modes_mask = np.real(split_system.element_epr([split_system.elements['resonator']])) > 0.5
        pf_modes_mask = np.real(split_system.element_epr([split_system.elements['Purcell-filter']])) > 0.5
        positive_frequency_mask = np.imag(w_split) > 0
        # take lowest frequency of these modes
        wr_split = np.min(w_split[np.logical_and(resonator_modes_mask, positive_frequency_mask)])
        wp_split = np.min(w_split[np.logical_and(pf_modes_mask, positive_frequency_mask)])

        assert np.any(np.abs(w_rpf - wr_split) < 1e6) # 1e-7 precision
        assert np.any(np.abs(w_rpf - wp_split) < 1e6) # 1e-7 precision


if __name__ == '__main__':
    unittest.main()
