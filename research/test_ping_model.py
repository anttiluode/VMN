"""Physics and observation-contract checks, with independent fixtures."""
import unittest

import numpy as np

import ping_model as pm


class PingModelTests(unittest.TestCase):
    def fixture(self):
        return pm.PhaseBank(np.array([[2*np.pi,0.],[0.,2*np.pi]]),
                            np.eye(2,dtype=complex),np.zeros((2,2),complex),
                            np.zeros((2,2),complex))

    def test_exact_flow_preserves_integrated_phase_after_radial_relaxation(self):
        # A quarter-turn must survive even when radius starts off its cycle.
        z = pm.uncoupled_step(np.array([[2.+0j]]),np.pi/2,1.)
        self.assertAlmostEqual(float(np.angle(z[0,0])),np.pi/2,places=13)
        radius2 = .5/(1+(.5/4-1)*np.exp(-1))
        self.assertAlmostEqual(float(abs(z[0,0])**2),radius2,places=13)

    def test_store_path_integrates_the_measured_displacement(self):
        bank = self.fixture()
        start = np.array([[.1,.2]])
        increments = np.array([[[.15,-.05]],[[.05,.1]]])
        result = pm.store_path(bank,start,increments,np.random.default_rng(2),sigma=0.)
        np.testing.assert_allclose(result,np.sqrt(.5)*np.exp(2j*np.pi*np.array([[.3,.25]])),atol=1e-13)

    def test_goal_kick_has_the_exact_arctangent_phase_response(self):
        bank = self.fixture()
        z = np.full((1,2),np.sqrt(.5),dtype=complex)
        record,diagnostic = pm.query(bank,z,np.array([[.25,0.]]),.1,
                                     np.random.default_rng(5),sigma=0.)
        response = np.angle(record.local_samples*np.conj(record.local_pre[:,None,:]))
        np.testing.assert_allclose(response,np.tile([np.arctan(.1),0.],(1,3,1)),atol=1e-13)
        self.assertGreater(float(np.linalg.norm(diagnostic.post-z)),0.)
        np.testing.assert_array_equal(z,np.full((1,2),np.sqrt(.5),complex))

    def test_zero_pulse_single_bank_retains_noise_that_twin_cancels(self):
        bank = self.fixture()
        z = np.full((5,2),np.sqrt(.5),dtype=complex)
        record,diagnostic = pm.query(bank,z,np.zeros((5,2)),0.,
                                     np.random.default_rng(7),sigma=.05)
        measured = np.angle(record.local_samples*np.conj(record.local_pre[:,None,:]))
        self.assertGreater(float(np.linalg.norm(measured)),.01)
        np.testing.assert_allclose(diagnostic.twin_phase,np.zeros((5,3,2)),atol=1e-14)

    def test_sum_ports_are_the_specified_linear_measurements(self):
        bank = pm.make_bank(17,8)
        z = np.sqrt(.5)*np.exp(1j*np.linspace(0.,1.,8))[None,:]
        record,_ = pm.query(bank,z,np.array([[.2,.3]]),.3,np.random.default_rng(4),sigma=0.)
        np.testing.assert_allclose(record.sum_pre,z@bank.sum_weights,atol=1e-14)
        np.testing.assert_allclose(record.sum_samples,record.local_samples@bank.sum_weights,atol=1e-14)

    def test_coherent_position_error_is_indistinguishable_from_valid_position(self):
        bank = self.fixture()
        start = np.array([[.2,.3]])
        increments = np.zeros((2,1,2))
        biased = pm.store_path(bank,start,increments,np.random.default_rng(1),sigma=0.,
                                velocity_bias=np.array([[.05,-.025]]))
        shifted = pm.store_path(bank,np.array([[.3,.25]]),increments,np.random.default_rng(9),sigma=0.)
        np.testing.assert_allclose(biased,shifted,atol=1e-13)

    def test_query_integrates_the_coupling_on_the_actual_trajectory(self):
        # Independent high-accuracy ODE solver catches frozen-J or wrong conjugacy.
        from scipy.integrate import solve_ivp
        bank = pm.make_bank(22,8)
        z = np.sqrt(.5)*np.exp(1j*np.linspace(.1,.8,8))[None,:]
        goal = np.array([[.3,.6]])
        kicked = z[0]+.3*np.sqrt(.5)*np.exp(1j*(bank.wavevectors@goal[0]))
        def rhs(t,q):
            return (.5-np.abs(q)**2)*q+.1*(bank.vortex_a@q+bank.vortex_b@q.conj())
        expected = solve_ivp(rhs,(0.,4.),kicked,method='DOP853',rtol=2e-12,atol=2e-13).y[:,-1]
        _,diagnostic = pm.query(bank,z,goal,.3,np.random.default_rng(3),sigma=0.,kappa=.1,dt=.03125)
        np.testing.assert_allclose(diagnostic.post[0],expected,rtol=1e-7,atol=1e-8)

    def test_invalid_physics_inputs_are_rejected(self):
        bank = self.fixture()
        z = np.full((1,2),np.sqrt(.5),complex)
        for amplitude in (-.1,np.nan):
            with self.assertRaises(ValueError):
                pm.query(bank,z,np.zeros((1,2)),amplitude,np.random.default_rng(1))
        with self.assertRaises(ValueError):
            pm.store_path(bank,np.zeros((1,2)),np.zeros((2,1,3)),np.random.default_rng(1))
        with self.assertRaises(ValueError):
            pm.uncoupled_step(z,0.,0.)


if __name__ == '__main__':
    unittest.main()
