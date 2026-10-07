"""Frozen, single-bank query-and-restore gate; see RESTORE_PROTOCOL.md."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from check_math import ROOT
from check_ping import array_hash,reader_metadata
from ping_model import MU,PhaseBank,make_bank,store_path,velocity_trajectory
from ping_readout import direct_features,fit_reader,local_features,phase_features,sum_features
from restore_model import paired_query


SEEDS = tuple(range(20263007,20263011))
COUNTS = (1024,256,512)
AMPLITUDES = (.05,.15,.3)
QUERY_NOISE = (0.,.05)
FAMILIES = ('fixed','radial_balance')
PATTERNS = ('repeat','mirror','opposite','single')
SPLITS = ('train','validation','test')
OFFSETS = (100000,200000,300000)
SOURCES = ('RESTORE_PROTOCOL.md','research/restore_model.py','research/check_restore.py',
           'research/test_restore_model.py','research/test_restore_experiment.py',
           'research/ping_model.py','research/ping_readout.py','research/check_ping.py',
           'research/check_math.py','research/geometry_lift.py')


def analytic_checks():
    bank = PhaseBank(np.array([[1.,0.],[1.,0.]]),np.eye(2,dtype=complex),
                     np.zeros((2,2),complex),np.zeros((2,2),complex))
    amplitudes = np.array([1e-4,3e-4,1e-3,3e-3,1e-2])
    z = np.full((1,2),np.sqrt(MU),complex)
    rows = []
    for phase in (.3,.7,1.1):
        values = []
        for a in amplitudes:
            _,d = paired_query(bank,z,np.array([[phase,0.]]),a,np.random.default_rng(1),sigma=0.)
            values.append(float(np.angle(d.post*np.conj(d.unpinged_post))[0,0]))
        slope = float(np.polyfit(np.log(amplitudes),np.log(np.abs(values)),1)[0])
        predicted = float(-(1-np.exp(-4))*np.sin(phase)*np.cos(phase))
        measured = values[0]/amplitudes[0]**2
        rows.append({'relative_phase':phase,'slope':slope,'residual_phase':values,
                     'second_order_coefficient':predicted,'measured_coefficient':measured,
                     'coefficient_relative_error':float(abs(measured-predicted)/abs(predicted))})
    mirrored = []
    for a in amplitudes:
        _,d = paired_query(bank,z,np.array([[np.cos(np.pi/6),1.]]),a,
                           np.random.default_rng(1),sigma=0.,pattern='mirror')
        mirrored.append(float(np.angle(d.post*np.conj(d.unpinged_post))[0,0]))
    mirror_slope = float(np.polyfit(np.log(amplitudes),np.log(np.abs(mirrored)),1)[0])
    predicted = float(2*np.sin(np.cos(np.pi/6)))
    off_cycle = {}
    for family in FAMILIES:
        values = []
        for a in amplitudes:
            _,d = paired_query(bank,z*.4,np.array([[.7,0.]]),a,np.random.default_rng(1),
                               sigma=0.,family=family)
            values.append(float(np.angle(d.post*np.conj(d.unpinged_post))[0,0]))
        off_cycle[family] = {'slope':float(np.polyfit(np.log(amplitudes),np.log(np.abs(values)),1)[0]),
                             'residual_phase':values}
    return {'amplitudes':amplitudes.tolist(),
            'ideal_opposite':{'passed':all(1.8<=r['slope']<=2.2 and r['coefficient_relative_error']<.003
                                          for r in rows),'rows':rows},
            'spatial_mirror_counterexample':{'passed':.95<=mirror_slope<=1.05,
                'slope':mirror_slope,'residual_phase':mirrored,
                'predicted_first_order_coefficient':predicted,
                'measured_first_order_coefficient':mirrored[0]/amplitudes[0]},
            'off_cycle_balance':{'passed':1.8<=off_cycle['radial_balance']['slope']<=2.2
                                 and .8<=off_cycle['fixed']['slope']<=1.2,**off_cycle}}


def run_bank(seed,units=32,counts=COUNTS,steps=200,amplitudes=AMPLITUDES,
             query_noise=QUERY_NOISE,pairs=8):
    bank = make_bank(seed,units)
    base,z = {},{}
    for split,count,offset in zip(SPLITS,counts,OFFSETS):
        start,velocity,end = velocity_trajectory(seed+offset,count,steps)
        rng = np.random.default_rng(seed+offset+41)
        angle = rng.uniform(0.,2*np.pi,count)
        radius = .35*np.sqrt(rng.uniform(0.,1.,count))
        displacement = radius[:,None]*np.c_[np.cos(angle),np.sin(angle)]
        base[split] = {'end':end,'goal':end+displacement,'displacement':displacement}
        z[split] = store_path(bank,start,velocity,np.random.default_rng(seed+offset+73))
    y = {s:base[s]['displacement'] for s in SPLITS}
    zero = float(np.linalg.norm(y['test'],axis=1).mean())
    position_reader = fit_reader(phase_features(z['train']),base['train']['end'],
                                 phase_features(z['validation']),base['validation']['end'])
    query_rows,repeated,controls = [],[],[{'seed':seed,'contract':'zero','error':zero,'reader':None}]

    def fit_score(features,contract,shuffle=False):
        ty,vy = y['train'],y['validation']
        if shuffle:
            ty = ty[np.random.default_rng(seed+700001).permutation(len(ty))]
            vy = vy[np.random.default_rng(seed+700002).permutation(len(vy))]
        reader = fit_reader(features['train'],ty,features['validation'],vy)
        pred = reader.predict(features['test'])
        error = float(np.linalg.norm(pred-y['test'],axis=1).mean())
        row = {'seed':seed,'units':units,'contract':contract,'error':error,
               'features':features['train'].shape[1],'reader':reader_metadata(reader),
               'training_feature_hash':array_hash(features['train']),
               'test_prediction_hash':array_hash(pred)}
        return reader,row

    _,row = fit_score({s:base[s]['goal'] for s in SPLITS},'goal_only')
    controls.append(row)
    _,row = fit_score({s:direct_features(bank,z[s],base[s]['goal']) for s in SPLITS},'direct_phase')
    controls.append(row)

    for family in FAMILIES:
        for amplitude in amplitudes:
            for sigma in query_noise:
                records = {}
                for s,offset in zip(SPLITS,OFFSETS):
                    records[s],_ = paired_query(bank,z[s],base[s]['goal'],amplitude,
                        np.random.default_rng(seed+offset+113),sigma=sigma,family=family)
                features = {s:local_features(records[s]) for s in SPLITS}
                reader,row = fit_score(features,'local_ports')
                row.update(family=family,amplitude=amplitude,query_noise=sigma,
                           output_real_channels=2*units,output_real_samples=8*units,
                           answer_latency=4.,completion_latency=8.,observed_banks=1)
                query_rows.append(row)
                if family=='fixed' and amplitude==.3 and sigma==.05:
                    _,control = fit_score(features,'shuffled_local',shuffle=True)
                    controls.append(control)
                    _,control = fit_score({s:sum_features(records[s],base[s]['goal']) for s in SPLITS},
                                           'four_channel_active')
                    controls.append(control)
                    passive = {s:paired_query(bank,z[s],base[s]['goal'],amplitude,
                        np.random.default_rng(seed+offset+113),sigma=sigma,pattern='passive')[0]
                        for s,offset in zip(SPLITS,OFFSETS)}
                    _,control = fit_score({s:sum_features(passive[s],base[s]['goal']) for s in SPLITS},
                                           'four_channel_passive')
                    controls.append(control)
                for pattern in PATTERNS:
                    pinged,reference = z['test'].copy(),z['test'].copy()
                    total_energy = np.zeros(counts[2])
                    for index in range(1,pairs+1):
                        record,d = paired_query(bank,pinged,base['test']['goal'],amplitude,
                            np.random.default_rng(seed+OFFSETS[-1]+113+4096*(index-1)),sigma=sigma,
                            family=family,pattern=pattern,reference=reference)
                        predicted = reader.predict(local_features(record))
                        query_error = float(np.linalg.norm(predicted-y['test'],axis=1).mean())
                        pinged,reference = d.post,d.unpinged_post
                        after = float(np.linalg.norm(position_reader.predict(phase_features(pinged))
                                                     -base['test']['end'],axis=1).mean())
                        passive_error = float(np.linalg.norm(position_reader.predict(phase_features(reference))
                                                             -base['test']['end'],axis=1).mean())
                        phase = np.angle(pinged*np.conj(reference))
                        total_energy += d.pulse_energy
                        repeated.append({'seed':seed,'units':units,'family':family,'pattern':pattern,
                            'amplitude':amplitude,'query_noise':sigma,'pairs':index,
                            'query_error':query_error,'position_error':after,
                            'unqueried_position_error':passive_error,'added_position_error':after-passive_error,
                            'causal_phase_rms':float(np.sqrt(np.mean(phase**2))),
                            'causal_state_change_rms':float(np.sqrt(np.mean(abs(pinged-reference)**2))),
                            'energy_proxy_per_pair':float(d.pulse_energy.mean()),
                            'total_energy_proxy_mean':float(total_energy.mean()),
                            'answer_time':8*(index-1)+4,'completion_time':8*index,
                            'reader_fit_hash':row['reader']['fit_array_hash']})
    return {'query_rows':query_rows,'repeated':repeated,'controls':controls,
            'metadata':{'seed':seed,'units':units,'counts':list(counts),'steps':steps,'observed_banks':1,
                'split_offsets':list(OFFSETS),'phase_code_hash':array_hash(bank.wavevectors),
                'sum_weights_hash':array_hash(bank.sum_weights),'state_scalars':2*units,
                'position_reader':reader_metadata(position_reader),
                'corpus_hashes':{s:array_hash(np.c_[base[s]['end'],base[s]['goal']]) for s in SPLITS}}}


def evaluate_gates(report):
    def selected(family,pattern):
        return sorted((r for r in report['repeated'] if r['units']==32 and r['family']==family
                       and r['pattern']==pattern and r['amplitude']==.3 and r['query_noise']==.05
                       and r['pairs']==8),key=lambda r:r['seed'])
    def control(contract):
        return np.array([r['error'] for r in sorted(report['controls'],key=lambda r:r['seed'])
                         if r['contract']==contract])
    def gate(passed,**values):
        return {'passed':bool(passed),**values}
    gates = {'ideal_opposite':report['analytic']['ideal_opposite'],
             'spatial_mirror_counterexample':report['analytic']['spatial_mirror_counterexample']}
    zero = control('zero')
    for family in FAMILIES:
        oppose,repeat = selected(family,'opposite'),selected(family,'repeat')
        first = np.array([r['error'] for r in sorted(report['query_rows'],key=lambda r:r['seed'])
                          if r['units']==32 and r['family']==family and r['amplitude']==.3
                          and r['query_noise']==.05])
        op,rep = np.array([r['causal_phase_rms'] for r in oppose]),np.array([r['causal_phase_rms'] for r in repeat])
        final = np.array([r['query_error'] for r in oppose])
        damage = np.array([r['added_position_error'] for r in oppose])
        half = op<=.5*rep
        gates['phase_reduction_'+family] = gate(op.mean()<=.5*rep.mean() and half.sum()>=3,
            opposite_phase_rms=float(op.mean()),repeat_phase_rms=float(rep.mean()),
            ratio=float(op.mean()/rep.mean()),half_reduction_wins=int(half.sum()))
        joint = half&(final<=.5*zero)&(final<=1.25*first)&(damage<=.005)
        gates['useful_repeated_'+family] = gate(op.mean()<=.5*rep.mean() and final.mean()<=.5*zero.mean()
            and final.mean()<=1.25*first.mean() and damage.mean()<=.005 and joint.sum()>=3,
            first_query_error=float(first.mean()),eighth_query_error=float(final.mean()),
            zero_error=float(zero.mean()),mean_added_position_error=float(damage.mean()),
            joint_wins=int(joint.sum()),per_bank_joint=joint.tolist())
    active,passive,goal = [control(c) for c in ('four_channel_active','four_channel_passive','goal_only')]
    joint = (active<passive)&(active<goal)
    gates['limited_port_query'] = gate(active.mean()<=.9*passive.mean() and active.mean()<=.9*goal.mean()
        and active.mean()<=.5*zero.mean() and joint.sum()>=3,error=float(active.mean()),
        passive_error=float(passive.mean()),goal_only_error=float(goal.mean()),joint_wins=int(joint.sum()))
    return gates


def summarize(report):
    groups = {}
    for row in report['repeated']:
        key = tuple(row[k] for k in ('family','pattern','amplitude','query_noise','pairs'))
        groups.setdefault(key,[]).append(row)
    summary = []
    for key,rows in groups.items():
        item = dict(zip(('family','pattern','amplitude','query_noise','pairs'),key))
        for metric in ('query_error','position_error','unqueried_position_error','added_position_error',
                       'causal_phase_rms','energy_proxy_per_pair','total_energy_proxy_mean'):
            values = np.array([r[metric] for r in rows])
            item[metric+'_mean'] = float(values.mean())
            item[metric+'_bank_sd'] = float(values.std())
        summary.append(item)
    return summary


def make_figure(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(14,4.6),constrained_layout=True)
    math = report['analytic']
    a = np.array(math['amplitudes'])
    axes[0].loglog(a,np.abs(math['ideal_opposite']['rows'][1]['residual_phase']),'o-',label='Opposite; on cycle')
    axes[0].loglog(a,np.abs(math['spatial_mirror_counterexample']['residual_phase']),'s-',label='Spatial mirror; forward unit')
    axes[0].loglog(a,np.abs(math['off_cycle_balance']['fixed']['residual_phase']),'^-',label='Opposite; off cycle')
    axes[0].loglog(a,np.abs(math['off_cycle_balance']['radial_balance']['residual_phase']),'d-',label='Balanced; off cycle')
    axes[0].set(xlabel='Pulse amplitude',ylabel='Remaining phase change (radians)',title='Noiseless mathematical checks')
    axes[0].legend(fontsize=8)
    for family,style in [('fixed','-'),('radial_balance','--')]:
        for pattern,color in [('repeat','#bd4937'),('mirror','#a58738'),('opposite','#16857c')]:
            rows = sorted((r for r in report['summary'] if r['family']==family and r['pattern']==pattern
                           and r['amplitude']==.3 and r['query_noise']==.05),key=lambda r:r['pairs'])
            x = [r['pairs'] for r in rows]
            label = pattern.capitalize()+(' (balanced)' if style=='--' else '')
            axes[1].plot(x,[r['causal_phase_rms_mean'] for r in rows],style,color=color,label=label)
            axes[2].plot(x,[r['query_error_mean'] for r in rows],style,color=color,label=label)
    axes[1].set(xlabel='Completed pulse pairs',ylabel='Causal phase RMS (radians)',title='Read disturbance; noisy bank')
    axes[1].legend(fontsize=8)
    axes[2].axhline(np.mean([r['error'] for r in report['controls'] if r['contract']=='zero']),
                    color='#555555',ls=':',label='Zero answer')
    axes[2].set(xlabel='Query number',ylabel='Mean goal-vector error',title='Frozen reader; pulse amplitude 0.3')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'restore_checks.json')
    args = parser.parse_args()
    started = time.perf_counter()
    report = {'setup':{'seeds':list(SEEDS),'units':32,'counts':list(COUNTS),'steps':200,
        'amplitudes':list(AMPLITUDES),'query_noise':list(QUERY_NOISE),'storage_noise':.05,
        'families':list(FAMILIES),'patterns':list(PATTERNS),'pairs':8,'dt':.25,
        'answer_time_within_pair':4.,'completion_time_per_pair':8.,'mirror_axis':'y=0.5'},
        'analytic':analytic_checks(),'query_rows':[],'repeated':[],'controls':[],'banks':[],
        'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    for seed in SEEDS:
        bank = run_bank(seed)
        for key in ('query_rows','repeated','controls'):
            report[key].extend(bank[key])
        report['banks'].append(bank['metadata'])
        print(f'bank {seed}: {time.perf_counter()-started:.1f}s',flush=True)
    report['gates'] = evaluate_gates(report)
    report['summary'] = summarize(report)
    report['elapsed_seconds'] = time.perf_counter()-started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    make_figure(report,args.output.with_name('restore_summary.svg'))
    print(json.dumps(report['gates'],indent=2,allow_nan=False))


if __name__=='__main__':
    main()
