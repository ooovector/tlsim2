import unittest
from tlsim2.tl import default_tl_basis, MultiTransmissionLine
from tlsim2.circuit import Circuit
from tlsim2.lumped import LumpedTwoTerminal
import numpy as np


class SplitTest(unittest.TestCase):
    def test_split_resonator_purcell(self):
        ll_coupler = [[476.595e-9, 169.262e-9],
                      [169.262e-9, 476.595e-9]]
        cl_coupler = [[166.304e-12, -59.062e-12],
                      [-59.062e-12, 166.304e-12]]
        ll = [[416.120e-9]]
        cl = [[166.448e-12]]

        sys_r = Circuit(name='r')
        sys_p = Circuit(name='p')
        sys_f = Circuit(name='f')

        t = 1
        t2 = 0

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

        rp = MultiTransmissionLine('rp', n=2, l=rp_length, ll=np.asarray(ll_coupler),
                                   cl=np.asarray(cl_coupler), basis=default_basis2)
        rp_r, rp_p, rp_rp = rp.split({'rp_r': ['i1', 'o1'], 'rp_p': ['i0', 'o0']})
        # rp = MultiTransmissionLine('rp', n=1, l=rp_length, ll=ll, cl=cl, basis=default_basis)

        pf = MultiTransmissionLine('pf', n=2, l=pf_length, ll=np.asarray(ll_coupler),
                                   cl=np.asarray(cl_coupler), basis=default_basis2)
        pf_p, pf_f, pf_pf = pf.split({'pf_p': ['i1', 'o1'], 'pf_f': ['i0', 'o0']})

        rs = MultiTransmissionLine('rs', n=1, l=rs_length, ll=ll, cl=cl, basis=default_basis)
        ro = MultiTransmissionLine('ro', n=1, l=ro_length, ll=ll, cl=cl, basis=default_basis)
        ps = MultiTransmissionLine('ps', n=1, l=ps_length, ll=ll, cl=cl, basis=default_basis)
        pm = MultiTransmissionLine('pm', n=1, l=pm_length, ll=ll, cl=cl, basis=default_basis)
        po = MultiTransmissionLine('po', n=1, l=po_length, ll=ll, cl=cl, basis=default_basis)

        cap = LumpedTwoTerminal(name='c', c=r_cap)
        p1 = LumpedTwoTerminal(name='p1', r=f_res)
        p2 = LumpedTwoTerminal(name='p2', r=f_res)

        sys_f.add_element(pf_f, {'i0': 1, 'o0': 2})
        sys_f.add_element(p1, {'i': 1, 'o': 0})
        sys_f.add_element(p2, {'i': 2, 'o': 0})  # notch-port-coupled resonator

        sys_p.add_element(ps, {'i0': 3, 'o0': 0})
        sys_p.add_element(pf_p, {'i1': 3, 'o1': 4})
        sys_p.add_element(pm, {'i0': 4, 'o0': 5})
        sys_p.add_element(rp_p, {'i0': 5, 'o0': 6})
        sys_p.add_element(po, {'i0': 6, 'o0': 9})

        sys_r.add_element(ro, {'i0': 7, 'o0': 10})
        sys_r.add_element(rp_r, {'i1': 7, 'o1': 8})
        sys_r.add_element(rs, {'i0': 8, 'o0': 0})
        sys_r.add_element(cap, {'i': 10, 'o': 0})

        sys_r.short(0)
        sys_p.short(0)
        sys_f.short(0)

        w_r, v_r, node_names_r = sys_r.compute_system_modes()
        w_p, v_p, node_names_p = sys_p.compute_system_modes()
        w_f, v_f, node_names_f = sys_f.compute_system_modes()

        sys_rpf_mod = Circuit()  # in principle, all connections by now are established, all that need
        # to be done now is reconcillation.
        sys_f_element = sys_f.make_element({1: 1, 2: 2}, cutoff_high=1e11)
        sys_p_element = sys_p.make_element({3: 3, 4: 4, 5: 5, 6: 6}, cutoff_high=1e11)
        sys_r_element = sys_r.make_element({7: 7, 8: 8}, cutoff_high=1e11)

        sys_rpf_mod.add_element(sys_f_element, {1: 1, 2: 2})
        sys_rpf_mod.add_element(sys_p_element, {3: 3, 4: 4, 5: 5, 6: 6})
        sys_rpf_mod.add_element(sys_r_element, {7: 7, 8: 8})

        sys_rpf_mod.add_element(rp_rp, {'i0': 5, 'o0': 6, 'i1': 7, 'o1': 8})
        sys_rpf_mod.add_element(pf_pf, {'i0': 1, 'o0': 2, 'i1': 3, 'o1': 4})

        w_rpf, v_rpf, node_names_rpf = sys_rpf_mod.compute_system_modes()

        frequencies_r = np.imag(w_r / (2 * np.pi)) / 1e9
        decays_r = np.real(w_r / 1e9) * 2
        mask_r = frequencies_r > 0
        decays_r = decays_r[mask_r]
        frequencies_r = frequencies_r[mask_r]

        frequencies_p = np.imag(w_p / (2 * np.pi)) / 1e9
        decays_p = np.real(w_p / 1e9) * 2
        mask_p = frequencies_p > 0
        decays_p = decays_p[mask_p]
        frequencies_p = frequencies_p[mask_p]

        frequencies_rpf = np.imag(w_rpf / (2 * np.pi)) / 1e9
        decays_rpf = np.real(w_rpf / 1e9) * 2
        mask_rpf = frequencies_rpf > 0
        decays_rpf = decays_rpf[mask_rpf]
        frequencies_rpf = frequencies_rpf[mask_rpf]

        print(np.round(frequencies_r, 3), np.round(decays_r, 3),
              np.round(frequencies_p, 3), np.round(decays_p, 3),
              np.round(frequencies_rpf, 3), np.round(decays_rpf, 3))

    # def test_tl_split(self):
    #     # self.assertEqual(True, False)  # add assertion here
    #     n = 2
    #     t = 0
    #
    #     ll_coupler = [[4, 1],
    #                   [1, 4]]
    #     cl_coupler = [[3, -1],
    #                   [-1, 3]]
    #     length = 1
    #
    #     default_basis2 = default_tl_basis(n, t)
    #     coupler = MultiTransmissionLine('Coupler', n=n, l=length, ll=ll_coupler, cl=cl_coupler, basis=default_basis2)
    #     r, l, rl = coupler.split({'r': ['i0', 'o0'], 'l': ['i1', 'o1']})
    #     li_init, c_init, ri_init = coupler.get_element_licri()
    #     sys1 = Circuit('SplitTest1')
    #     sys1.add_element(coupler, {'i0': 'i0', 'o0': 'o0', 'i1': 'i1', 'o1': 'o1'})
    #
    #     print('Splitting with intrinsic (omitted) dofs, "rr": ', r.terminals_a, r.terminals_b)
    #     print('Splitting with intrinsic (omitted) dofs, "ll": ', l.terminals_a, l.terminals_b)
    #     print('Splitting with intrinsic (omitted) dofs, "rl": ', rl.terminals_a, rl.terminals_b)
    #
    #     sys2 = Circuit('SplitTest2')
    #     sys2.add_element(r, {'i0': 'i0', 'o0': 'o0'})
    #     sys2.add_element(l, {'i1': 'i1', 'o1': 'o1'})
    #     sys2.add_element(rl, {'i0': 'i0', 'o0': 'o0', 'i1': 'i1', 'o1': 'o1'})
    #
    #     # li, c, ri, node_names = sys1.get_system_licri()
    #     frequencies1, modes1, node_names1 = sys1.compute_system_modes()
    #     frequencies2, modes2, node_names2 = sys2.compute_system_modes()
    #     print(frequencies1, frequencies2)
    #
    #     # print (li, li_init)
    #     # and now re-create the element


if __name__ == '__main__':
    unittest.main()
