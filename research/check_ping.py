"""Reproduce the predeclared single-bank and limited-port phase-memory gate."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from check_math import ROOT
from ping_model import make_bank,query,store_path,velocity_trajectory
from ping_readout import (direct_features,fit_reader,local_features,
                          phase_features,sum_features)


SEEDS = tuple(range(20262007,20262011))
COUNTS = (1024,256,512)
UNITS = (8,32)
SPLITS = ('train','validation','test')
OFFSETS = (100000,200000,300000)
SOURCES = ('PING_PROTOCOL.md','research/ping_model.py','research/ping_readout.py',
           'research/check_ping.py','research/test_ping_model.py',
           'research/test_ping_readout.py','research/test_ping_experiment.py',
           'research/check_math.py','research/geometry_lift.py')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def reader_metadata(reader):
    data = reader.weights if reader.kind=='ridge' else reader.training_features
    return {'kind':reader.kind,'parameter':reader.parameter,
            'validation_error':reader.validation_error,
            'candidate_validation':reader.candidate_validation,
            'storage_scalars':reader.storage_scalars,
            'storage_bytes_float64':8*reader.storage_scalars,
            'normalization_hash':array_hash(np.r_[reader.mean,reader.scale]),
            'fit_array_hash':array_hash(data)}


def run_bank(seed,units,counts=COUNTS,steps=200):
    bank = make_bank(seed,units)
    base = {}
    for split,count,offset in zip(SPLITS,counts,OFFSETS):
        start,velocity,end = velocity_trajectory(seed+offset,count,steps)
        rng = np.random.default_rng(seed+offset+41)
        angle = rng.uniform(0.,2*np.pi,count)
        radius = .35*np.sqrt(rng.uniform(0.,1.,count))
        displacement = radius[:,None]*np.c_[np.cos(angle),np.sin(angle)]
        base[split] = {'start':start,'velocity':velocity,'end':end,
                       'displacement':displacement,'goal':end+displacement}
    rows,memory,repeated = [],[],[]
    targets = {s:base[s]['displacement'] for s in SPLITS}
    zero_error = float(np.linalg.norm(targets['test'],axis=1).mean())

    def score(features,scenario,contract,condition,channels,amplitude=0.,kappa=0.,shuffle=False):
        train_y,val_y = targets['train'],targets['validation']
        if shuffle:
            train_y = train_y[np.random.default_rng(seed+700001).permutation(len(train_y))]
            val_y = val_y[np.random.default_rng(seed+700002).permutation(len(val_y))]
        reader = fit_reader(features['train'],train_y,features['validation'],val_y)
        prediction = reader.predict(features['test'])
        errors = np.linalg.norm(prediction-targets['test'],axis=1)
        sample_times = 1 if contract in ('goal_only','direct_phase') else 3 if contract=='privileged_twin' else 4
        rows.append({'seed':seed,'units':units,'scenario':scenario,'contract':contract,
                     'condition':condition,'amplitude':amplitude,'kappa':kappa,
                     'features':features['train'].shape[1],
                     'output_real_channels':channels,'output_real_samples':channels*sample_times,
                     'observed_banks':2 if contract=='privileged_twin' else 0 if contract=='goal_only' else 1,
                     'pulse_pattern_real_values':2*units if amplitude>0 else 0,
                     'listen_time':4. if contract in ('local_ports','four_channel','privileged_twin') else 0.,
                     'training_episodes':counts[0],
                     'validation_episodes':counts[1],'test_episodes':counts[2],
                     'error':float(errors.mean()),'episode_error_sd':float(errors.std()),
                     'zero_answer_error':zero_error,'reader':reader_metadata(reader),
                     'training_feature_hash':array_hash(features['train']),
                     'test_prediction_hash':array_hash(prediction)})
        return reader

    for scenario in ('independent','sensor_bias'):
        z,bias,observed_position = {},{},{}
        for split,offset in zip(SPLITS,OFFSETS):
            entry = base[split]
            bias[split] = (np.random.default_rng(seed+offset+101).normal(0.,.0003,entry['start'].shape)
                           if scenario=='sensor_bias' else np.zeros_like(entry['start']))
            z[split] = store_path(bank,entry['start'],entry['velocity'],
                                 np.random.default_rng(seed+offset+73),velocity_bias=bias[split])
            observed_position[split] = entry['start']+np.sum(entry['velocity']+bias[split],axis=0)
        digital = base['test']['goal']-observed_position['test']
        errors = np.linalg.norm(digital-targets['test'],axis=1)
        rows.append({'seed':seed,'units':units,'scenario':scenario,'contract':'digital',
                     'condition':'measured_velocity','amplitude':0.,'kappa':0.,'features':2,
                     'output_real_channels':2,'output_real_samples':2,'observed_banks':0,
                     'pulse_pattern_real_values':0,'listen_time':0.,'training_episodes':counts[0],
                     'validation_episodes':counts[1],'test_episodes':counts[2],
                     'error':float(errors.mean()),'episode_error_sd':float(errors.std()),
                     'zero_answer_error':zero_error,'reader':None,
                     'test_prediction_hash':array_hash(digital)})
        score({s:base[s]['goal'] for s in SPLITS},scenario,'goal_only','no_measurement',0)
        score({s:direct_features(bank,z[s],base[s]['goal']) for s in SPLITS},
              scenario,'direct_phase','reference',2*units)
        position_reader = fit_reader(phase_features(z['train']),base['train']['end'],
                                      phase_features(z['validation']),base['validation']['end'])
        position_error_before = float(np.linalg.norm(
            position_reader.predict(phase_features(z['test']))-base['test']['end'],axis=1).mean())

        # Each condition sees the same initial histories and noise draws. A twin
        # is used only by its labelled diagnostic or the causal damage calculation.
        for amplitude,kappa,condition in ((0.,0.,'passive'),(.05,0.,'pulse_0.05'),
                                           (.3,0.,'pulse_0.3'),(.3,.1,'query_vortex')):
            records,diagnostics = {},{}
            for split,offset in zip(SPLITS,OFFSETS):
                records[split],diagnostics[split] = query(bank,z[split],base[split]['goal'],
                    amplitude,np.random.default_rng(seed+offset+113),kappa=kappa)
            if amplitude>0:
                local = {s:local_features(records[s]) for s in SPLITS}
                score(local,scenario,'local_ports',condition,2*units,amplitude,kappa)
            if kappa==0:
                summed = {s:sum_features(records[s],base[s]['goal']) for s in SPLITS}
                score(summed,scenario,'four_channel',condition,4,amplitude)
            if amplitude==.3 and kappa==0:
                score({s:diagnostics[s].twin_phase.reshape(counts[i],-1)
                       for i,s in enumerate(SPLITS)},scenario,'privileged_twin',condition,
                       4*units,.3)
                score(local,scenario,'local_ports','shuffled_targets',2*units,.3,shuffle=True)
                score(summed,scenario,'four_channel','shuffled_targets',4,.3,shuffle=True)
            if amplitude>0:
                diagnostic = diagnostics['test']
                after = float(np.linalg.norm(position_reader.predict(phase_features(diagnostic.post))
                                             -base['test']['end'],axis=1).mean())
                control = float(np.linalg.norm(position_reader.predict(phase_features(diagnostic.unpinged_post))
                                               -base['test']['end'],axis=1).mean())
                memory.append({'seed':seed,'units':units,'scenario':scenario,'condition':condition,
                               'before_error':position_error_before,'after_error':after,
                               'same_time_unpinged_error':control,'added_error':after-control,
                               'causal_phase_change_rms':float(np.sqrt(np.mean(diagnostic.twin_phase[:,-1]**2))),
                               'position_reader':reader_metadata(position_reader)})

        pinged,reference = z['test'].copy(),z['test'].copy()
        rng = np.random.default_rng(seed+810001)
        for index in range(1,5):
            angle = rng.uniform(0.,2*np.pi,counts[2])
            radius = .35*np.sqrt(rng.uniform(0.,1.,counts[2]))
            goals = base['test']['end']+radius[:,None]*np.c_[np.cos(angle),np.sin(angle)]
            _,diagnostic = query(bank,pinged,goals,.3,rng,reference=reference)
            pinged,reference = diagnostic.post,diagnostic.unpinged_post
            after = float(np.linalg.norm(position_reader.predict(phase_features(pinged))
                                         -base['test']['end'],axis=1).mean())
            control = float(np.linalg.norm(position_reader.predict(phase_features(reference))
                                           -base['test']['end'],axis=1).mean())
            repeated.append({'seed':seed,'units':units,'scenario':scenario,'queries':index,
                             'elapsed_listen_time':4*index,'after_error':after,
                             'same_time_unpinged_error':control,'added_error':after-control})

    return {'rows':rows,'memory':memory,'repeated':repeated,
            'metadata':{'seed':seed,'units':units,'counts':list(counts),'steps':steps,
                        'split_offsets':list(OFFSETS),'phase_code_hash':array_hash(bank.wavevectors),
                        'sum_weights_hash':array_hash(bank.sum_weights),
                        'vortex_hash':array_hash(np.stack([bank.vortex_a,bank.vortex_b])),
                        'state_float64_scalars':2*units,'sum_weight_float64_scalars':4*units,
                        'input_wavevector_float64_scalars':2*units,
                        'vortex_coefficient_float64_scalars':4*units*units}}


def evaluate_gates(report):
    def errors(contract,condition,units=32,scenario='independent'):
        rows = sorted((r for r in report['rows'] if r['contract']==contract
                       and r['condition']==condition and r['units']==units and r['scenario']==scenario),
                      key=lambda r:r['seed'])
        return np.array([r['error'] for r in rows])
    physical = errors('local_ports','pulse_0.3')
    direct = errors('direct_phase','reference')
    summed = errors('four_channel','pulse_0.3')
    passive = errors('four_channel','passive')
    goal = errors('goal_only','no_measurement')
    vortex = errors('local_ports','query_vortex')
    zero = np.array([r['zero_answer_error'] for r in sorted(report['rows'],key=lambda r:r['seed'])
                     if r['contract']=='local_ports' and r['condition']=='pulse_0.3'
                     and r['units']==32 and r['scenario']=='independent'])
    memories = [m for m in report['memory'] if m['units']==32 and m['scenario']=='independent']
    pulse_memory = sorted((m for m in memories if m['condition']=='pulse_0.3'),key=lambda m:m['seed'])
    vortex_memory = sorted((m for m in memories if m['condition']=='query_vortex'),key=lambda m:m['seed'])
    def entry(passed,**values):
        return {'passed':bool(passed),**values}
    damage = np.array([m['added_error'] for m in pulse_memory])
    extra = np.array([v['after_error']-p['after_error'] for v,p in zip(vortex_memory,pulse_memory)])
    return {
        'single_bank_query':entry(physical.mean()<=.5*zero.mean() and np.sum(physical<zero)>=3,
                                 error=float(physical.mean()),zero_error=float(zero.mean()),wins=int(np.sum(physical<zero))),
        'near_direct_access':entry(physical.mean()<=1.25*direct.mean() and np.sum(physical<=1.25*direct)>=3,
                                  ratio=float(physical.mean()/direct.mean()),per_bank_ratio=(physical/direct).tolist()),
        'limited_port_query':entry(summed.mean()<=.9*passive.mean() and summed.mean()<=.9*goal.mean()
                                  and np.sum((summed<passive)&(summed<goal))>=3
                                  and summed.mean()<=.5*zero.mean(),
                                  error=float(summed.mean()),passive_error=float(passive.mean()),
                                  goal_only_error=float(goal.mean()),wins_vs_passive=int(np.sum(summed<passive)),
                                  wins_vs_goal_only=int(np.sum(summed<goal)),
                                  joint_wins=int(np.sum((summed<passive)&(summed<goal)))),
        'read_disturbance':entry(damage.mean()<=.005,mean_added_error=float(damage.mean()),per_bank_added_error=damage.tolist()),
        'vortex_value':entry(vortex.mean()<=.9*physical.mean() and np.sum(vortex<physical)>=3 and extra.mean()<=.005,
                             vortex_error=float(vortex.mean()),uncoupled_error=float(physical.mean()),
                             wins=int(np.sum(vortex<physical)),mean_extra_position_error=float(extra.mean())),
        'redundancy':entry(direct.mean()<=.75*errors('direct_phase','reference',8).mean(),
                           error_32=float(direct.mean()),error_8=float(errors('direct_phase','reference',8).mean()))}


def summarize(report):
    groups = {}
    for row in report['rows']:
        key = (row['units'],row['scenario'],row['contract'],row['condition'])
        groups.setdefault(key,[]).append(row)
    summary = []
    for (units,scenario,contract,condition),rows in groups.items():
        error = np.array([r['error'] for r in rows])
        summary.append({'units':units,'scenario':scenario,'contract':contract,'condition':condition,
                        'error_mean':float(error.mean()),'bank_error_sd':float(error.std()),
                        'features':rows[0]['features'],'output_real_channels':rows[0]['output_real_channels'],
                        'decoder_storage_scalars':[r['reader']['storage_scalars'] if r['reader'] else 0 for r in rows],
                        'selected_readers':[r['reader']['kind'] if r['reader'] else 'arithmetic' for r in rows]})
    return summary


def make_figure(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig,axes = plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    groups = report['summary']
    methods = [('digital','measured_velocity','Digital'),('direct_phase','reference','Direct phases'),
               ('privileged_twin','pulse_0.3','Privileged twin'),('local_ports','pulse_0.3','Single bank'),
               ('four_channel','pulse_0.3','Four channels'),('four_channel','passive','Passive ports'),
               ('goal_only','no_measurement','Goal only')]
    errors,spreads = [],[]
    for contract,condition,_ in methods:
        row = next(r for r in groups if r['units']==32 and r['scenario']=='independent'
                   and r['contract']==contract and r['condition']==condition)
        errors.append(row['error_mean']);spreads.append(row['bank_error_sd'])
    axes[0].barh(np.arange(len(methods)),errors,xerr=spreads,color=['#596575','#16857c','#aa8647','#237fc2','#8866b5','#bbb3cc','#929ca9'])
    axes[0].set_yticks(np.arange(len(methods)),[m[2] for m in methods])
    axes[0].invert_yaxis();axes[0].set_xlabel('Mean goal-vector error (box width = 1)')
    axes[0].set_title('32 units; independent internal noise')
    colors = ['#16857c','#237fc2','#596575']
    for index,(contract,condition,label) in enumerate((methods[1],methods[3],methods[0])):
        for scenario,style in [('independent','-'),('sensor_bias','--')]:
            values = [next(r['error_mean'] for r in groups if r['units']==n and r['scenario']==scenario
                           and r['contract']==contract and r['condition']==condition) for n in UNITS]
            axes[1].plot(UNITS,values,style+'o',color=colors[index],label=label+(' + sensor bias' if style=='--' else ''))
    axes[1].set_xticks(UNITS);axes[1].set_xlabel('Complex units');axes[1].set_ylabel('Mean goal-vector error')
    axes[1].set_title('Redundancy and coherent sensor error');axes[1].legend(fontsize=8)
    for units,color in [(8,'#aa8647'),(32,'#237fc2')]:
        values = [np.mean([r['added_error'] for r in report['repeated'] if r['units']==units
                          and r['scenario']=='independent' and r['queries']==k]) for k in range(1,5)]
        axes[2].plot(range(1,5),values,'o-',color=color,label=f'{units} units')
    axes[2].axhline(.005,color='#929ca9',linestyle='--',label='One-query gate threshold')
    axes[2].set_xlabel('Successive queries (4 time units each)');axes[2].set_ylabel('Added position error vs same-time unpinged copy')
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True));axes[2].set_title('Query disturbance');axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(output/'ping_summary.svg')
    fig.savefig('/tmp/vmn_ping_summary.png',dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args = parser.parse_args()
    began = time.perf_counter()
    report = {'setup':{'seeds':list(SEEDS),'units':list(UNITS),'counts':list(COUNTS),
                       'steps':200,'sigma':.05,'mu':.5,'dt':.25,'listen_times':[1,2,4],
                       'sensor_bias_sd_per_step':.0003,'goal_radius':.35,
                       'primary_amplitude':.3,'query_vortex_strength':.1},
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
              'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
              'rows':[],'memory':[],'repeated':[],'banks':[]}
    for units in UNITS:
        for seed in SEEDS:
            start = time.perf_counter()
            result = run_bank(seed,units)
            for name in ('rows','memory','repeated'):
                report[name].extend(result[name])
            report['banks'].append(result['metadata'])
            print(f'{units} units, bank {seed}: {time.perf_counter()-start:.1f}s',flush=True)
    report['summary'] = summarize(report)
    report['gates'] = evaluate_gates(report)
    report['wall_seconds'] = time.perf_counter()-began
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'ping_checks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    make_figure(report,args.output_dir)
    print(json.dumps(report['gates'],indent=2),flush=True)


if __name__=='__main__':
    main()
