"""Checks for lag alignment, training-only fitting and state-code decoding."""
import importlib
import unittest

import numpy as np


def gate_module():
    try:
        return importlib.import_module("check_geometry")
    except ModuleNotFoundError as exc:
        raise AssertionError("The held-out geometry gate is not implemented yet") from exc


class ReadoutAndCodeTests(unittest.TestCase):
    def test_targets_use_past_inputs_and_orthogonal_polynomial_orders(self):
        gate = gate_module()
        idx, targets, descriptions = gate.memory_targets(np.array([-1.,0.,1.,.5,-.5]),lags=(1,2))
        np.testing.assert_array_equal(idx,[2,3,4])
        np.testing.assert_allclose(targets[0],[0.,-1.,-.5,1.,0.,-1.])
        self.assertEqual(descriptions[0],{"order":1,"lag":1})
        self.assertEqual(descriptions[-1],{"order":3,"lag":2})

    def test_affine_readout_uses_training_statistics_with_constant_features(self):
        gate = gate_module()
        x = np.column_stack([np.arange(-2,3),np.ones(5)])
        y = (2*x[:,0]+3)[:,None]
        fitted = gate.fit_readout(x,y,0.)
        np.testing.assert_allclose(fitted.mean,[0.,1.])
        saved_mean = fitted.mean.copy()
        np.testing.assert_allclose(fitted.predict(np.array([[10.,1.]])),[[23.]],atol=1e-12)
        np.testing.assert_allclose(fitted.mean,saved_mean)

    def test_test_targets_do_not_select_the_readout(self):
        gate = gate_module()
        x = np.arange(-10,11,dtype=float)[:,None]
        y = 3*x+2
        readout, penalty = gate.select_readout(x,y,x[::2],y[::2],penalties=(1e-6,1.))
        self.assertEqual(penalty,1e-6)
        self.assertLess(np.mean((readout.predict(x)-y)**2),1e-6)

    def test_signed_score_does_not_credit_a_wrong_constant_prediction(self):
        gate = gate_module()
        result = gate.score_predictions(np.array([[-1.],[0.],[1.]]),np.full((3,1),5.))
        self.assertLess(result["r2"][0],0.)
        self.assertEqual(result["correlation_squared"][0],0.)

    def test_state_code_selects_current_difference_blocks_without_event_labels(self):
        gate = gate_module()
        base = np.arange(6,dtype=float)
        actual = base+np.array([.1,0.,.05,.02,0.,.2])
        code = gate.encode_state_difference(base,actual,1)
        np.testing.assert_array_equal(code["indices"],[2])
        np.testing.assert_allclose(code["offsets"],[[0.,.2]])
        expected = base+np.array([0.,0.,0.,0.,0.,.2])
        np.testing.assert_allclose(gate.decode_state_difference(base,code),expected)
        full = gate.encode_state_difference(base,actual,3)
        np.testing.assert_allclose(gate.decode_state_difference(base,full),actual)

    def test_lag_and_code_validation_rejects_invalid_or_nonfinite_data(self):
        gate = gate_module()
        for lags in [(0,),(1.5,),(10,)]:
            with self.assertRaises(ValueError):
                gate.memory_targets(np.ones(5),lags=lags)
        for k in [0,4,1.5]:
            with self.assertRaises(ValueError):
                gate.encode_state_difference(np.zeros(6),np.ones(6),k)
        with self.assertRaises(ValueError):
            gate.fit_readout(np.array([[0.],[np.nan]]),np.zeros((2,1)),.1)


if __name__ == "__main__":
    unittest.main()
