"""Reader tests pin parity, quadratic decoding and held-out selection."""
import importlib
import unittest

import numpy as np

from geometry_memory import fit_readout, score_predictions


def reader_module():
    try:
        return importlib.import_module("geometry_reader")
    except ModuleNotFoundError as exc:
        raise AssertionError("The quadratic reader is not implemented yet") from exc


class QuadraticReaderTests(unittest.TestCase):
    def test_features_include_the_state_and_all_local_quadratic_coordinates(self):
        reader = reader_module()
        x = np.array([[2.,3.,4.,-1.]])
        np.testing.assert_allclose(reader.reader_features(x,"quadratic"),
                                  [[2.,3.,4.,-1.,13.,17.,-5.,15.,12.,-8.]])
        np.testing.assert_array_equal(reader.reader_features(x,"raw"),x)

    def test_added_coordinates_are_even_under_history_sign_reversal(self):
        reader = reader_module()
        x = np.array([[.2,-.3,.4,.1],[-.2,.8,.1,-.5]])
        f = reader.reader_features(x,"quadratic")
        reversed_f = reader.reader_features(-x,"quadratic")
        np.testing.assert_allclose(reversed_f[:,:4],-f[:,:4])
        np.testing.assert_allclose(reversed_f[:,4:],f[:,4:])

    def test_quadratic_reader_recovers_an_even_target_on_unseen_amplitudes(self):
        reader = reader_module()
        u = np.r_[np.linspace(.2,.95,8),-np.linspace(.2,.95,8)]
        v = np.r_[np.linspace(.1,.9,7),-np.linspace(.1,.9,7)]
        x,xt = np.column_stack([u,np.zeros(len(u))]),np.column_stack([v,np.zeros(len(v))])
        y,yt = ((3*u*u-1)/2)[:,None],((3*v*v-1)/2)[:,None]
        raw = fit_readout(reader.reader_features(x,"raw"),y,0.)
        quad = fit_readout(reader.reader_features(x,"quadratic"),y,0.)
        self.assertLessEqual(score_predictions(yt,raw.predict(xt))["r2"][0],0.)
        self.assertGreater(score_predictions(yt,quad.predict(reader.reader_features(xt,"quadratic")))["r2"][0],.999999999)

    def test_only_test_changes_leave_fitted_parameters_and_penalties_unchanged(self):
        reader = reader_module()
        datasets = []
        for n in [30,24,20]:
            u = np.linspace(-1,1,n)
            datasets.append((np.column_stack([u,np.zeros(n)]),
                np.column_stack([u,(3*u*u-1)/2,(5*u**3-3*u)/2])))
        descriptions = [{"order":i,"lag":1} for i in (1,2,3)]
        original = reader.compare_readers(datasets,descriptions)
        self.assertIsNone(original["raw"]["test"]["mean_long_lag_r2"])
        changed = datasets[:2]+[(datasets[2][0]*2,datasets[2][1]+100.)]
        second = reader.compare_readers(changed,descriptions)
        for mode in ("raw","quadratic"):
            self.assertEqual(original[mode]["ridge_penalty"],second[mode]["ridge_penalty"])
            self.assertEqual(original[mode]["parameters_sha256"],second[mode]["parameters_sha256"])
            self.assertNotEqual(original[mode]["test"],second[mode]["test"])

    def test_feature_validation_rejects_invalid_states_or_reader_names(self):
        reader = reader_module()
        for x in [np.ones((2,3)),np.empty((0,2)),np.array([[0.,np.nan]]),np.array([[1j,0.]])]:
            with self.assertRaises(ValueError):
                reader.reader_features(x,"quadratic")
        with self.assertRaises(ValueError):
            reader.reader_features(np.zeros((2,2)),"unknown")

    def test_shuffled_fitting_labels_destroy_known_even_target_decoding(self):
        reader = reader_module()
        rng = np.random.default_rng(17)
        datasets = []
        for n in (800,300,400):
            u = rng.uniform(-1,1,n)
            datasets.append((np.column_stack([u,np.zeros(n)]),
                np.column_stack([u,(3*u*u-1)/2,(5*u**3-3*u)/2])))
        descriptions = [{"order":i,"lag":1} for i in (1,2,3)]
        normal = reader.compare_readers(datasets,descriptions)
        shuffled = reader.compare_readers(datasets,descriptions,shuffle_seed=91)
        self.assertGreater(normal["quadratic"]["test"]["mean_order_r2"]["2"],.99)
        self.assertLess(shuffled["quadratic"]["test"]["mean_order_r2"]["2"],.05)
        self.assertNotEqual(normal["quadratic"]["parameters_sha256"],
                            shuffled["quadratic"]["parameters_sha256"])

    def test_fresh_dataset_alignment_uses_independent_past_inputs(self):
        reader = reader_module()
        from geometry_model import make_model
        lengths,washout,seed = (32,24,16),40,12345
        data,descriptions,hashes = reader.fresh_datasets(make_model(7),seed,lengths,washout)
        rng = np.random.default_rng(seed)
        expected_inputs = [rng.uniform(-1,1,washout+n) for n in lengths]
        for (states,targets),inputs,n in zip(data,expected_inputs,lengths):
            self.assertEqual(states.shape,(n,16))
            self.assertEqual(targets.shape,(n,18))
            self.assertAlmostEqual(targets[0,0],inputs[washout-1])
            self.assertAlmostEqual(targets[0,5],inputs[washout-32])
        self.assertEqual(descriptions[6],{"order":2,"lag":1})
        self.assertEqual(len(set(hashes)),3)

    def test_primary_gate_requires_gain_seed_wins_and_positive_score(self):
        reader = reader_module()
        rows = [{"seed":i,"variant":"fixed","readers":{
            "raw":{"test":{"mean_order_r2":{"2":0.}}},
            "quadratic":{"test":{"mean_order_r2":{"2":value}}}}}
            for i,value in enumerate([.04,.04,.04,-.02])]
        result = reader.reader_gate(rows,"fixed")
        self.assertTrue(result["passed_predeclared_gate"])
        self.assertAlmostEqual(result["mean_difference"],.025)
        self.assertEqual(result["wins"],3)
        for row in rows:
            row["readers"]["raw"]["test"]["mean_order_r2"]["2"] -= .1
            row["readers"]["quadratic"]["test"]["mean_order_r2"]["2"] -= .1
        self.assertFalse(reader.reader_gate(rows,"fixed")["passed_predeclared_gate"])


if __name__ == "__main__":
    unittest.main()
