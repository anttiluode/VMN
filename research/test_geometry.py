"""Independent checks for the geometry/shear gate; no external services."""
import importlib
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from check_math import derivative_fd, relative_error


def model_module():
    try:
        return importlib.import_module("geometry_model")
    except ModuleNotFoundError as exc:
        raise AssertionError("The matched geometry model is not implemented yet") from exc


class GeometryMathTests(unittest.TestCase):
    def test_regularized_velocity_has_the_physical_two_vortex_value(self):
        gm = model_module()
        q = np.array([-.5, 0., .5, 0.])
        expected = np.array([0., -1/(2*np.pi*1.04), 0., 1/(2*np.pi*1.04)])
        np.testing.assert_allclose(gm.vortex_velocity(q, np.ones(2), delta=.2), expected)

    def test_regularized_jacobian_matches_velocity_derivatives(self):
        gm = model_module()
        q = np.array([-.6, .1, .7, -.2, .15, .9])
        gamma = np.array([.8, -1.2, .6])
        fd = derivative_fd(lambda x: gm.vortex_velocity(x, gamma, delta=.2), q, 1e-5)
        self.assertLess(relative_error(gm.vortex_jacobian(q, gamma, delta=.2), fd), 1e-8)

    def test_geometry_changes_higher_order_terms_but_matches_response_at_rest(self):
        gm = model_module()
        fixed = gm.make_model(seed=7, geometry=0, beta=2)
        moving = gm.make_model(seed=7, geometry=1, beta=2)
        zero = np.zeros(16)
        np.testing.assert_allclose(fixed.vector_field(zero), zero, atol=1e-15)
        np.testing.assert_allclose(moving.vector_field(zero), zero, atol=1e-15)
        np.testing.assert_allclose(fixed.jacobian(zero), moving.jacobian(zero), atol=1e-14)
        q = np.linspace(-.25, .25, 16)
        self.assertGreater(np.linalg.norm(fixed.vector_field(q)-moving.vector_field(q)), 1e-5)

    def test_hybrid_jacobian_includes_the_derivative_of_moving_geometry(self):
        gm = model_module()
        q = np.linspace(-.21, .17, 16)
        for geometry in (0, 1):
            for beta in (0, 2):
                model = gm.make_model(seed=7, geometry=geometry, beta=beta)
                fd = derivative_fd(lambda x: model.vector_field(x, .2), q, 1e-5)
                self.assertLess(relative_error(model.jacobian(q), fd), 2e-8)

    def test_boundary_and_rollout_match_a_known_linear_oscillator(self):
        gm = model_module()
        model = gm.GeometryModel(anchors=np.zeros((1,2)), gamma=np.ones(1),
            omega=np.array([1.3]), decay=np.array([.4]), input_vector=np.zeros(2),
            coupling=0., mu=.35, beta=0., geometry=0., cubic=0.)
        self.assertAlmostEqual(model.stability_boundary()["mu_critical"], .4, places=12)
        q0 = np.array([.2, -.1])
        coarse = gm.rollout(model, np.zeros(80), .25, initial=q0)[-1]
        observed = gm.rollout(model, np.zeros(80), .25, substeps=4, initial=q0)[-1]
        expected = complex(*q0)*np.exp((-.05+1.3j)*20)
        self.assertLess(np.linalg.norm(observed-[expected.real,expected.imag]), 6e-6)
        self.assertLess(np.linalg.norm(observed-[expected.real,expected.imag]),
                        np.linalg.norm(coarse-[expected.real,expected.imag])/10)

    def test_exact_phase_kick_matches_independent_polar_integration(self):
        gm = model_module()
        mu, beta = .1, 2.
        rstar = np.sqrt(mu)
        rho = .1*rstar
        expected = -.19062035960864973
        self.assertAlmostEqual(gm.phase_kick(mu,beta,rho), expected, places=13)
        def rhs(t,y):
            r = y[0]
            return [mu*r-r**3, -beta*(r*r-mu)]
        sol = solve_ivp(rhs,(0,140),[rstar+rho,0.],method="DOP853",rtol=1e-12,atol=1e-13)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(sol.y[1,-1],expected,places=10)

    def test_phase_memory_can_be_invisible_to_the_instantaneous_jacobian(self):
        gm = model_module()
        model = gm.GeometryModel(anchors=np.zeros((1,2)),gamma=np.ones(1),
            omega=np.ones(1),decay=np.zeros(1),input_vector=np.zeros(2),
            coupling=0.,mu=.1,beta=2.,geometry=0.)
        q = np.array([np.sqrt(.1),0.])
        np.testing.assert_allclose(model.jacobian(q),model.jacobian(-q),atol=1e-14)
        self.assertGreater(np.linalg.norm(model.vector_field(q)-model.vector_field(-q)),.1)
        self.assertAlmostEqual(gm.relaxed_response_norm(.1,2,np.pi),0.,places=12)
        angle = np.pi/6
        changed = np.sqrt(.1)*np.array([np.cos(angle),np.sin(angle)])
        measured = np.linalg.norm(model.jacobian(changed)-model.jacobian(q),2)
        self.assertAlmostEqual(measured,.22360679774997896,places=12)
        self.assertAlmostEqual(gm.relaxed_response_norm(.1,2,angle),measured,places=12)

    def test_invalid_phase_and_geometry_parameters_are_rejected(self):
        gm = model_module()
        for mu,beta,rho in [(0.,2.,.1),(-.1,2.,.1),(.1,2.,-1.)]:
            with self.assertRaises(ValueError):
                gm.phase_kick(mu,beta,rho)
        with self.assertRaises(ValueError):
            gm.make_model(seed=7,geometry=-1)
        with self.assertRaises(ValueError):
            gm.make_model(seed=7,beta=np.nan)


if __name__ == "__main__":
    unittest.main()
