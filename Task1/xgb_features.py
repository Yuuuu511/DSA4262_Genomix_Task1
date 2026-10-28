"""Streaming, site-level features; no identifiers or labels are model inputs."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 1
MOTIFS = [d+r+'AC'+h for d in 'AGT' for r in 'AG' for h in 'ACT']
SIGNALS = [f'{p}_{s}' for p in ['minus1', 'center', 'plus1'] for s in ['dwell', 'signal_sd', 'current']]
FEATURE_NAMES = ([f'{s}_{stat}' for stat in ['mean','std','q10','median','q90'] for s in SIGNALS]
    + [f'{p}_logdwell_{stat}' for stat in ['mean','std'] for p in ['minus1','center','plus1']]
    + [f'contrast_{p}_{s}_{stat}' for stat in ['mean','std'] for p in ['minus1','plus1'] for s in ['dwell','signal_sd','current']]
    + ['log1p_reads'] + [f'base{i}_{b}' for i in range(7) for b in 'ACGT']
    + [f'motif_{m}' for m in MOTIFS])

def site_features(sequence, reads):
    a = np.asarray(reads, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 9 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Each site must contain finite reads with nine features')
    sequence = sequence.upper().replace('U','T')
    if len(sequence) != 7 or any(b not in 'ACGT' for b in sequence):
        raise ValueError(f'Invalid seven-base sequence: {sequence}')
    if (a[:,[0,3,6]] < 0).any():
        raise ValueError('Negative dwell time')
    q = np.quantile(a, [.1,.5,.9], axis=0)
    logs = np.log1p(a[:,[0,3,6]] * 1000)
    diffs = np.concatenate([a[:,3:6]-a[:,:3],a[:,3:6]-a[:,6:9]],axis=1)
    return np.concatenate([a.mean(0), a.std(0), q.ravel(), logs.mean(0),logs.std(0),
        diffs.mean(0), diffs.std(0), [np.log1p(len(a))],
        [float(base==b) for base in sequence for b in 'ACGT'],
        [float(sequence[1:6]==m) for m in MOTIFS]]).astype(np.float32)

def iter_sites(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line_no,line in enumerate(f,1):
            obj = json.loads(line)
            for transcript, positions in obj.items():
                for position, seqs in positions.items():
                    if len(seqs)!=1: raise ValueError(f'Expected one sequence at line {line_no}')
                    sequence, reads = next(iter(seqs.items()))
                    yield transcript,int(position),sequence,reads

def load_features(path, cache_dir=None):
    path=Path(path).resolve()
    cache=None
    if cache_dir:
        stat=path.stat()
        token=hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}:{VERSION}'.encode()).hexdigest()[:20]
        cache=Path(cache_dir)/f'{token}.npz'
        if cache.exists():
            with np.load(cache,allow_pickle=False) as z:
                return pd.DataFrame({'transcript_id':z['transcripts'],'transcript_position':z['positions']}),z['X']
    ids=[];positions=[];features=[]
    for i,(tx,pos,seq,reads) in enumerate(iter_sites(path),1):
        ids.append(tx);positions.append(pos);features.append(site_features(seq,reads))
        if i%20000==0: print(f'{path.name}: extracted {i:,} sites',flush=True)
    meta=pd.DataFrame({'transcript_id':ids,'transcript_position':positions})
    if meta.duplicated().any():raise ValueError('Duplicate transcript-position keys')
    X=np.asarray(features,dtype=np.float32)
    if X.shape!=(len(meta),len(FEATURE_NAMES)):raise ValueError('Feature schema mismatch')
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,transcripts=np.asarray(ids),positions=np.asarray(positions),X=X)
    return meta,X

def read_labels(path,meta):
    labels=pd.read_csv(path)
    keys=['transcript_id','transcript_position']
    if labels.duplicated(keys).any():raise ValueError('Duplicate metadata keys')
    joined=meta.merge(labels,on=keys,how='left',validate='one_to_one',indicator=True)
    if not (joined['_merge']=='both').all() or len(labels)!=len(meta):
        raise ValueError('Metadata does not exactly match JSON sites')
    return joined.drop(columns='_merge')
