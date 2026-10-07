"""The later probe must measure dynamics, including the linear negative control."""
import importlib
import unittest

import numpy as np

from check_math import relative_error, tangent_trajectory, trajectory
from geometry_model import make_model


class ResponseTests(unittest.TestCase):
    def test_fixed_linear_history_does_not_change_the_response_operator(self):
        module = importlib.import_module("geometry_response")
        self.assertTrue(hasattr(module,"response_records"),"response_records is not implemented yet")
        model = make_model(seed=9,geometry=0,beta=0,cubic=0)
        record = module.response_records(model,np.linspace(-.1,.1,16),
            [{"site":3,"write":[.07,0.]}],1.,np.array([.2,.4]),(1,))
        baseline = record["baseline_response"]
        update = record["events"][0]["response_update"]
        self.assertLess(np.linalg.norm(update)/np.linalg.norm(baseline),2e-8)

    def test_analytic_tangent_predicts_an_independent_finite_pulse(self):
        model = make_model(seed=9,geometry=1,beta=2)
        q = np.linspace(-.1,.1,16)
        times = np.array([.2,.5])
        rhs = model.vector_field
        _,phi = tangent_trajectory(rhs,model.jacobian,q,times)
        probe = np.random.default_rng(93).normal(size=16)
        probe /= np.linalg.norm(probe)
        eps = 1e-5
        fd = (trajectory(rhs,q+eps*probe,times)-trajectory(rhs,q-eps*probe,times))/2/eps
        expected = np.einsum("tij,j->ti",phi,probe)
        self.assertLess(relative_error(fd,expected),1e-7)


if __name__ == "__main__":
    unittest.main()
