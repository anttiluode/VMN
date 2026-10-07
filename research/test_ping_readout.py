"""Checks catch hidden-twin use and leakage across data splits."""
import unittest

import numpy as np

import ping_model as pm
import ping_readout as pr


class PingReadoutTests(unittest.TestCase):
    def test_local_features_use_the_pre_pulse_measurement_not_a_twin(self):
        bank = pm.make_bank(9,8)
        z = np.full((10,8),np.sqrt(.5),complex)
        record,_ = pm.query(bank,z,np.zeros((10,2)),0.,np.random.default_rng(3))
        features = pr.local_features(record)
        self.assertEqual(features.shape,(10,24))
        self.assertGreater(float(np.linalg.norm(features)),.01)

    def test_sum_features_expose_only_four_channels_and_known_goal(self):
        bank = pm.make_bank(3,8)
        goals = np.array([[.2,.3],[.1,.4]])
        record,_ = pm.query(bank,np.full((2,8),np.sqrt(.5),complex),goals,.3,np.random.default_rng(2))
        altered = pm.PortRecord(record.local_pre*20,record.local_samples*10,
                                record.sum_pre,record.sum_samples)
        first,second = pr.sum_features(record,goals),pr.sum_features(altered,goals)
        self.assertEqual(first.shape,(2,18))
        np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(first[:,-2:],goals)

    def test_reader_training_scale_does_not_use_validation_or_test(self):
        x = np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.],[5.,5.]])
        y = np.c_[x[:,0],-x[:,1]]
        fitted = pr.fit_reader(x,y,np.array([[1000.,1000.]]),np.array([[1000.,-1000.]]))
        np.testing.assert_allclose(fitted.mean,[2.5,2.5])
        np.testing.assert_allclose(fitted.scale,[np.sqrt(35/12),np.sqrt(35/12)])
        np.testing.assert_allclose(fitted.predict(np.array([[2.5,2.5]])),[[2.5,-2.5]],atol=.01)

    def test_reader_matches_an_unseen_linear_mapping(self):
        x = np.linspace(-2.,2.,80)[:,None]
        y = np.c_[2*x[:,0]+1,-x[:,0]+.5]
        model = pr.fit_reader(x,y,np.array([[-1.3],[.7]]),np.array([[-1.6,1.8],[2.4,-.2]]))
        np.testing.assert_allclose(model.predict(np.array([[.25]])),[[1.5,.25]],atol=.002)
        self.assertEqual(model.kind,'ridge')
        self.assertGreater(model.storage_scalars,0)

    def test_reader_rejects_nonfinite_or_misaligned_training_rows(self):
        with self.assertRaises(ValueError):
            pr.fit_reader(np.array([[np.nan]]),np.zeros((1,2)),np.ones((1,1)),np.zeros((1,2)))
        with self.assertRaises(ValueError):
            pr.fit_reader(np.ones((2,1)),np.zeros((1,2)),np.ones((1,1)),np.zeros((1,2)))


if __name__ == '__main__':
    unittest.main()
