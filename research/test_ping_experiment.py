"""A small end-to-end fixture catches reference, split and reproducibility errors."""
import unittest

import numpy as np

import check_ping as cp


class PingExperimentTests(unittest.TestCase):
    def test_limited_port_gate_requires_joint_wins_on_the_same_banks(self):
        rows,memory = [],[]
        for i,seed in enumerate(range(4)):
            for contract,condition,error in (
                ('local_ports','pulse_0.3',.04),('direct_phase','reference',.02),
                ('four_channel','pulse_0.3',.08),
                ('four_channel','passive',[.12,.12,.07,.12][i]),
                ('goal_only','no_measurement',[.12,.12,.12,.07][i]),
                ('local_ports','query_vortex',.05)):
                rows.append({'seed':seed,'units':32,'scenario':'independent',
                             'contract':contract,'condition':condition,'error':error,
                             'zero_answer_error':.236})
            rows.append({'seed':seed,'units':8,'scenario':'independent',
                         'contract':'direct_phase','condition':'reference','error':.06})
            for condition in ('pulse_0.3','query_vortex'):
                memory.append({'seed':seed,'units':32,'scenario':'independent',
                               'condition':condition,'after_error':.03,'added_error':.002})
        gate = cp.evaluate_gates({'rows':rows,'memory':memory})['limited_port_query']
        self.assertFalse(gate['passed'])

    def test_small_experiment_preserves_exact_digital_reference_and_replays(self):
        first = cp.run_bank(31,8,counts=(32,8,12),steps=8)
        second = cp.run_bank(31,8,counts=(32,8,12),steps=8)
        rows = first['rows']
        for scenario in ('independent','sensor_bias'):
            selected = [r for r in rows if r['scenario']==scenario]
            self.assertEqual(len(selected),12)
        digital = [r for r in rows if r['contract']=='digital']
        self.assertLess(digital[0]['error'],1e-12)
        self.assertGreater(digital[1]['error'],.0001)
        self.assertEqual(first,second)
        self.assertTrue(all(np.isfinite(r['error']) for r in rows))
        self.assertTrue(all(r['training_episodes']==32 for r in rows))
        summed = [r for r in rows if r['contract']=='four_channel']
        self.assertTrue(all(r['features']==18 for r in summed))
        self.assertTrue(all(r['output_real_channels']==4 for r in summed))


if __name__ == '__main__':
    unittest.main()
