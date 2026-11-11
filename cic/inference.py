"""Anonymous end-to-end target inference: no labels, M/K, or k selection API."""
from pathlib import Path
from time import perf_counter
import numpy as np

from .preprocessing import compute_relation_split_stats
from .point_relations import prepare_points
from .trajectory_encoder import make_features, encode_all
from .retrieval import candidates
from .scoring import score_candidates
from .calibration import calibrated_probabilities
from .partition import partition_probabilities
from .device import sync, reset_peak_memory, memory_snapshot
from .evaluation import ResourceMonitor
from .io import log, heartbeat, write_json


def infer(trajectories, encoder, scorer, support, calibrator, cfg, kp, kc, device, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    reset_peak_memory(device)
    times = {}
    with ResourceMonitor() as monitor:
        t = perf_counter()
        stats = compute_relation_split_stats(trajectories)
        prepared = prepare_points(trajectories,stats,cfg['max_points'])
        times['poi_statistics_and_point_preparation'] = perf_counter()-t
        t = perf_counter()
        embeddings = encode_all(encoder,make_features(prepared),device)
        sync(device)
        times['trajectory_encoding'] = perf_counter()-t
        t = perf_counter()
        with heartbeat('target_fixed_k_retrieval'):
            pairs,origin = candidates(trajectories,stats['poi_idf'],embeddings,kp,kc,cfg['similarity_block_rows'])
        times['poi_vectors_and_dual_retrieval'] = perf_counter()-t
        log('target_candidates_ready',kp=kp,kc=kc,trajectories=len(trajectories),edges=len(pairs))
        t = perf_counter()
        logits,scoring = score_candidates(prepared,pairs,scorer,device,cfg)
        times['point_relations_and_mlp'] = perf_counter()-t
        graph = dict(n=len(trajectories),pairs=pairs,embeddings=embeddings,logits=logits)
        t = perf_counter()
        with heartbeat('target_fourteen_features_and_probability'):
            p,feature_info = calibrated_probabilities(graph,trajectories,support,calibrator,cfg)
        times['fourteen_features_and_probability'] = perf_counter()-t
        t = perf_counter()
        with heartbeat('target_observed_edge_grouping'):
            pred,grouping = partition_probabilities(len(trajectories),pairs,p)
        times['grouping'] = perf_counter()-t
        sync(device)
    # Freeze the entire prediction before the caller is allowed to evaluate it.
    keys = np.asarray([t.key for t in trajectories])
    t = perf_counter()
    np.savez_compressed(output/'candidates.npz',pairs=pairs,origin=origin,embeddings=embeddings,trajectory_keys=keys)
    np.save(output/'candidate_logits.npy',logits)
    np.save(output/'candidate_identity_probabilities.npy',p)
    np.savez_compressed(output/'partition.npz',pred=pred,trajectory_keys=keys)
    saving = perf_counter()-t
    n = len(trajectories)
    all_pairs = n*(n-1)//2
    sizes = np.unique(pred,return_counts=True)[1]
    report = dict(trajectories=n,candidate_pairs=len(pairs),possible_pairs=all_pairs,
        pair_reduction_fraction=1-len(pairs)/all_pairs if all_pairs else 0.,
        preset_kp=kp,preset_kc=kc,effective_kp=min(kp,max(0,n-1)),effective_kc=min(kc,max(0,n-1)),
        clusters=len(sizes),singletons=int(np.sum(sizes==1)),largest_cluster=int(sizes.max()),
        stages_seconds=times,inference_seconds=monitor.as_dict()['stage_seconds'],
        save_seconds=saving,resources=monitor.as_dict(),device_memory=memory_snapshot(device),
        timing_scope='includes POI statistics, point preparation, encoding, retrieval, relations, MLP, all fourteen features, calibration, grouping; excludes loading, training, saving, evaluation',
        scoring=scoring,features=feature_info,grouping=grouping,
        uses_M=False,uses_K=False,uses_identity_labels=False,additional_pairs=0,missing_edges='unknown; no likelihood/cost term')
    write_json(output/'inference.json',report)
    return pred,report
