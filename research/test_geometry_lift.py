"""Independent physical and algebraic expectations for the complex tangent."""
import importlib
import unittest

import numpy as np

from check_math import derivative_fd
from geometry_model import make_model


def lift_module():
    try:
        return importlib.import_module("geometry_lift")
    except ModuleNotFoundError as exc:
        raise AssertionError("The full complex tangent is not implemented yet") from exc


class ComplexTangentTests(unittest.TestCase):
    def test_real_block_separates_rotation_and_conjugate_response(self):
        lift = lift_module()
        a,b = lift.real_to_complex(np.array([[2.,3.],[-5.,7.]]))
        np.testing.assert_allclose(a,[[4.5-4j]])
        np.testing.assert_allclose(b,[[-2.5-1j]])
        a,b = lift.real_to_complex(np.array([[0.,-3.],[3.,0.]]))
        np.testing.assert_allclose(a,[[3j]])
        np.testing.assert_allclose(b,[[0.]])

    def test_lift_maps_physical_perturbations_and_preserves_conjugacy(self):
        lift = lift_module()
        transform = lift.unitary_lift(2)
        np.testing.assert_allclose(transform@transform.conj().T,np.eye(4),atol=1e-15)
        real = np.array([1.,2.,-3.,4.])
        expected = np.array([1+2j,-3+4j,1-2j,-3-4j])/np.sqrt(2)
        np.testing.assert_allclose(transform@real,expected)
        j = np.array([[2.,3.,1.,0.],[-5.,7.,0.,2.],[0.,1.,-.2,-1.],[2.,0.,1.,-.2]])
        a,b = lift.real_to_complex(j)
        doubled = lift.doubled_matrix(a,b)
        np.testing.assert_allclose(doubled@expected,transform@(j@real),atol=1e-14)
        action = doubled@expected
        np.testing.assert_allclose(action[2:],action[:2].conj(),atol=1e-14)

    def test_analytic_smoothed_tangent_matches_physical_finite_differences(self):
        lift = lift_module()
        q = np.linspace(-.21,.17,16)
        transform = lift.unitary_lift(8)
        for geometry in (0.,1.):
            for beta in (0.,2.):
                model = make_model(7,geometry,beta)
                a,b = lift.model_complex_tangent(model,q)
                fd = derivative_fd(lambda state:model.vector_field(state,.2),q,1e-5)
                np.testing.assert_allclose(lift.doubled_matrix(a,b),
                    transform@fd@transform.conj().T,rtol=1e-7,atol=2e-9)

    def test_two_vortex_smoothing_creates_an_ordinary_rotation_term(self):
        lift = lift_module()
        pos = np.array([[0.,0.],[1.,0.]])
        a,b = lift.vortex_complex_tangent(pos,np.ones(2),.2)
        laplacian = np.array([[1.,-1.],[-1.,1.]])
        np.testing.assert_allclose(a,1j*.04/(2*np.pi*1.04**2)*laplacian)
        np.testing.assert_allclose(b,-1j/(2*np.pi*1.04**2)*laplacian)
        a,b = lift.vortex_complex_tangent(pos,np.ones(2),0.)
        np.testing.assert_allclose(a,0.,atol=1e-15)
        np.testing.assert_allclose(b,-1j/(2*np.pi)*laplacian)

    def test_complex_interfaces_reject_malformed_or_nonfinite_arrays(self):
        lift = lift_module()
        for value in [np.eye(3),np.ones((2,3)),np.array([[np.nan,0.],[0.,1.]])]:
            with self.assertRaises(ValueError):
                lift.real_to_complex(value)
        with self.assertRaises(ValueError):
            lift.doubled_matrix(np.eye(2),np.eye(1))
        with self.assertRaises(ValueError):
            lift.vortex_complex_tangent(np.zeros((2,2)),np.ones(2),0.)


if __name__ == "__main__":
    unittest.main()
