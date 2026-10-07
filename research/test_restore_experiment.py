"""End-to-end and gate checks for the frozen restoration experiment."""
import unittest

import numpy as np

from check_restore import analytic_checks,evaluate_gates,run_bank


class RestoreExperimentTests(unittest.TestCase):
    def test_mathematical_checks_distinguish_mirroring_and_opposition(self):
        # Confusing mirror targets with negative waveforms flips these slopes.
        checks = analytic_checks()
        self.assertTrue(checks['ideal_opposite']['passed'])
        self.assertTrue(checks['spatial_mirror_counterexample']['passed'])
        self.assertTrue(checks['off_cycle_balance']['passed'])
        self.assertGreater(checks['spatial_mirror_counterexample']['slope'],.95)
        self.assertLess(checks['spatial_mirror_counterexample']['slope'],1.05)

    def test_small_run_replays_without_refitting_repeated_readers(self):
        # Refitting on test states or using fresh noise on controls breaks replay.
        kwargs = {'units':4,'counts':(20,8,10),'steps':4,'amplitudes':(.3,),
                  'query_noise':(.05,),'pairs':2}
        a,b = run_bank(17,**kwargs),run_bank(17,**kwargs)
        self.assertEqual(a,b)
        self.assertEqual(len(a['query_rows']),2)
        self.assertEqual(len(a['repeated']),16)
        for row in a['repeated']:
            query = next(q for q in a['query_rows'] if q['family']==row['family'])
            self.assertEqual(row['reader_fit_hash'],query['reader']['fit_array_hash'])
            if row['pairs']==1:
                self.assertEqual(row['query_error'],query['error'])
            if row['pattern']!='single':
                self.assertAlmostEqual(row['energy_proxy_per_pair'],.36,places=12)
        self.assertEqual(a['metadata']['observed_banks'],1)

    def test_useful_reads_require_joint_per_bank_success(self):
        # Different seed subsets cannot be pooled into a spurious joint win.
        queries,repeated,controls = [],[],[]
        for seed in range(4):
            controls.append({'seed':seed,'contract':'zero','error':.3})
            for family in ('fixed','radial_balance'):
                queries.append({'seed':seed,'units':32,'family':family,'amplitude':.3,
                                'query_noise':.05,'error':.04})
                for pattern in ('repeat','opposite'):
                    repeated.append({'seed':seed,'units':32,'family':family,'amplitude':.3,
                                     'query_noise':.05,'pattern':pattern,'pairs':8,
                                     'causal_phase_rms':.4 if pattern=='repeat' else .1,
                                     'query_error':.04 if seed!=0 else .08,
                                     'added_position_error':.008 if seed==3 else .001})
            for contract,error in (('four_channel_active',.05),('four_channel_passive',.2),
                                    ('goal_only',.2)):
                controls.append({'seed':seed,'contract':contract,'error':error})
        report = {'query_rows':queries,'repeated':repeated,'controls':controls,
                  'analytic':analytic_checks()}
        gates = evaluate_gates(report)
        for family in ('fixed','radial_balance'):
            gate = gates['useful_repeated_'+family]
            self.assertFalse(gate['passed'])
            self.assertEqual(gate['joint_wins'],2)
        self.assertTrue(gates['limited_port_query']['passed'])


if __name__=='__main__':
    unittest.main()
