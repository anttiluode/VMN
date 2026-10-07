"""Checks for the observable two-pulse controller, not biological analogies."""
import unittest

import numpy as np

from ping_model import MU,PhaseBank,make_bank,query
from ping_readout import local_features
from restore_model import paired_query,pulse_pair


def simple_bank():
    return PhaseBank(np.array([[1.,0.],[0.,1.]]),np.eye(2,dtype=complex),
                     np.zeros((2,2),complex),np.zeros((2,2),complex))


class RestoreModelTests(unittest.TestCase):
    def test_spatial_reflection_preserves_forward_waveform(self):
        # Conjugating every waveform instead of reflecting spatial y is wrong.
        first,second = pulse_pair(simple_bank(),np.full((1,2),np.sqrt(MU)),
                                   np.array([[.2,.7]]),.3,'fixed','mirror')
        np.testing.assert_allclose(first,.3*np.sqrt(MU)*np.exp(1j*np.array([[.2,.7]])))
        np.testing.assert_allclose(second,.3*np.sqrt(MU)*np.exp(1j*np.array([[.2,.3]])))
        self.assertEqual(first[0,0],second[0,0])

    def test_balanced_family_has_same_declared_energy_at_each_state(self):
        # Omitting normalization makes balanced controllers consume more energy.
        bank = simple_bank()
        z = np.array([[.2,1.2],[.4,.7]],complex)
        for family in ('fixed','radial_balance'):
            for pattern in ('repeat','mirror','opposite'):
                first,second = pulse_pair(bank,z,np.array([[.2,.7],[.4,.3]]),.3,family,pattern)
                energy = np.sum(abs(first)**2+abs(second)**2,axis=1)
                np.testing.assert_allclose(energy,[.18,.18],rtol=1e-13)

    def test_ideal_opposite_pair_has_derived_second_order_residual(self):
        # Using the same signed second pulse leaves a first-order phase shift.
        bank = simple_bank()
        z = np.full((1,2),np.sqrt(MU),complex)
        goals = np.array([[.7,1.1]])
        _,diagnostic = paired_query(bank,z,goals,1e-4,np.random.default_rng(1),sigma=0.)
        coefficient = -(1-np.exp(-4))*np.sin(goals)*np.cos(goals)
        measured = np.angle(diagnostic.post*np.conj(diagnostic.unpinged_post))/1e-8
        np.testing.assert_allclose(measured,coefficient,rtol=.001)

    def test_radial_balance_cancels_off_cycle_first_order_term(self):
        # A constant counterpulse fails when the baseline radius changes.
        bank = simple_bank()
        z = np.array([[.25,1.]],complex)
        goal = np.array([[.7,1.1]])
        residual = []
        for amplitude in (1e-3,1e-4):
            _,d = paired_query(bank,z,goal,amplitude,np.random.default_rng(1),
                                sigma=0.,family='radial_balance')
            residual.append(np.linalg.norm(np.angle(d.post*np.conj(d.unpinged_post))))
        self.assertGreater(residual[0]/residual[1],90.)
        self.assertLess(residual[0]/residual[1],110.)

    def test_second_pulse_cannot_change_already_collected_answer(self):
        # Sampling after compensation would leak different information to readers.
        bank = make_bank(8,4)
        z = np.full((3,4),np.sqrt(MU),complex)
        goals = np.array([[.2,.3],[.4,.8],[.6,.1]])
        records = [paired_query(bank,z,goals,.3,np.random.default_rng(3),pattern=p)[0]
                   for p in ('repeat','mirror','opposite')]
        for record in records[1:]:
            np.testing.assert_array_equal(local_features(record),local_features(records[0]))
        original,_ = query(bank,z,goals,.3,np.random.default_rng(3))
        np.testing.assert_array_equal(original.local_samples,records[0].local_samples)

    def test_counterfactual_changes_diagnostics_not_measured_record(self):
        # Allowing the twin into the listener would remove physical query noise.
        bank = make_bank(3,4)
        z = np.full((2,4),np.sqrt(MU),complex)
        goals = np.array([[.2,.3],[.4,.8]])
        a,_ = paired_query(bank,z,goals,.3,np.random.default_rng(2))
        b,_ = paired_query(bank,z,goals,.3,np.random.default_rng(2),reference=z*1j)
        np.testing.assert_array_equal(local_features(a),local_features(b))

    def test_passive_has_zero_causal_change_with_future_noise(self):
        # Different noise streams or unequal elapsed time would create fake damage.
        bank = make_bank(3,4)
        z = np.full((2,4),np.sqrt(MU),complex)
        _,d = paired_query(bank,z,np.array([[.2,.3],[.4,.8]]),.3,
                           np.random.default_rng(2),pattern='passive')
        np.testing.assert_array_equal(d.post,d.unpinged_post)
        np.testing.assert_array_equal(d.pulse_energy,[0.,0.])

    def test_controller_rejects_undefined_or_invalid_inputs(self):
        bank = simple_bank()
        z = np.ones((1,2),complex)
        g = np.zeros((1,2))
        for args in ((z,g,-.1,'fixed','repeat'),(z,g,.1,'bad','repeat'),
                     (z,g,.1,'fixed','bad'),(z[:,:1],g,.1,'fixed','repeat'),
                     (z,np.array([[np.nan,0.]]),.1,'fixed','repeat'),
                     (np.zeros_like(z),g,.1,'radial_balance','opposite')):
            with self.assertRaises(ValueError):
                pulse_pair(bank,*args)
        for kwargs in ({'sigma':-1.},{'dt':.3},{'reference':z[:,:1]}):
            with self.assertRaises(ValueError):
                paired_query(bank,z,g,.1,np.random.default_rng(1),**kwargs)


if __name__=='__main__':
    unittest.main()
