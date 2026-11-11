"""MLP training, fourteen-feature calibration and fixed-budget test inference."""
from pathlib import Path
from time import perf_counter
import argparse
import gc
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cic.io import read_json,write_json,resolve,sha256,log,heartbeat,code_fingerprint
from cic.protocol import load_protocol,save_protocol,validate_protocol,validate_neural_checkpoint
from cic.training import train_mlp
from cic.trajectory_encoder import TrajectoryEncoder,make_features,encode_all
from cic.point_relations import prepare_points
from cic.preprocessing import compute_relation_split_stats
from cic.retrieval import candidates
from cic.scoring import score_candidates
from cic.calibration import fit_calibration
from cic.inference import infer
from cic.evaluation import evaluate_partition,evaluate_candidates_after_prediction,ResourceMonitor
from cic.device import select_device,environment_info

def run_city(city,spec,cfg,protocol,base,dest,device):
    dest.mkdir(parents=True,exist_ok=False)
    begin = perf_counter()
    csv = resolve(base,spec['data'])
    if sha256(csv) != spec['data_sha256']:
        raise ValueError(f'{city}: dataset differs from the frozen preset configuration')
    encoder_path = resolve(base,spec['encoder'])
    if sha256(encoder_path) != spec['encoder_sha256']:
        raise ValueError(f'{city}: encoder changed after budgets were set')
    if cfg['seed'] != spec['encoder_seed'] or cfg['max_points'] != spec['encoder_max_points']:
        raise ValueError('Seed/point preprocessing differs from the frozen encoder preparation')
    roles,manifest = load_protocol(csv,protocol)
    original_manifest = read_json(resolve(base,spec['source_manifest']))
    if manifest != original_manifest:
        raise ValueError(f'{city}: source/validation/test split differs from preparation')
    kp,kc = spec['preset_kp'],spec['preset_kc']
    if not all(type(k) is int and k >= 0 for k in (kp,kc)):
        raise ValueError('Main requires two preset nonnegative integer budgets')
    save_protocol(dest,roles,manifest)
    write_json(dest/'configuration.json',dict(city=city,preset_kp=kp,preset_kc=kc,
                training=cfg,protocol=protocol,encoder_sha256=spec['encoder_sha256'],code_sha256=code_fingerprint()))
    checkpoint = torch.load(encoder_path,map_location='cpu',weights_only=True)
    validate_neural_checkpoint(checkpoint,manifest,cfg)
    encoder = TrajectoryEncoder().to(device)
    encoder.load_state_dict(checkpoint['model'],strict=True)
    encoder.eval()
    with ResourceMonitor() as training_resources:
        with heartbeat(f'{city}_fresh_pointwise_mlp'):
            scorer = train_mlp(roles['neural_train'],roles['val'],cfg,dest/'mlp',device)
    mlp_checkpoint = torch.load(dest/'mlp/pointwise_mlp.pt',map_location='cpu',weights_only=True)
    validate_neural_checkpoint(mlp_checkpoint,manifest,cfg)
    calibration = roles['calibration']
    with ResourceMonitor() as calibration_resources:
        with heartbeat(f'{city}_source_calibration_graph'):
            # A standalone graph over the independent 10% ORIGINAL TRAINING users.
            # Validation users/points/statistics are not part of this graph.
            stats = compute_relation_split_stats(calibration.trajectories)
            prepared = prepare_points(calibration.trajectories,stats,cfg['max_points'])
            embeddings = encode_all(encoder,make_features(prepared),device)
            pairs,origin = candidates(calibration.trajectories,stats['poi_idf'],embeddings,kp,kc,cfg['similarity_block_rows'])
            n = len(calibration.trajectories)
            graph_info = dict(scope='standalone independent original-training calibration users',
                users=len(calibration.users),trajectories=n,preset_kp=kp,preset_kc=kc,
                effective_kp=min(kp,n-1),effective_kc=min(kc,n-1),
                observed_edges=len(pairs),possible_pairs=n*(n-1)//2,
                observed_edge_density=len(pairs)/(n*(n-1)//2),
                poi_statistics_scope='this calibration cohort only; no neural-training/validation/test statistics',
                no_rescaling_of_k=True,missing_edges='unknown; no fabricated labels or scores')
            log('calibration_candidates_ready',**graph_info)
            logits,scoring = score_candidates(prepared,pairs,scorer,device,cfg,label='source_calibration_scoring')
            graph = dict(n=n,pairs=pairs,logits=logits,embeddings=embeddings)
            support,calibrator = fit_calibration(graph,calibration,cfg,dest/'calibration')
            write_json(dest/'calibration/graph.json',graph_info)
            write_json(dest/'calibration/scoring.json',scoring)
            np.savez_compressed(dest/'calibration/observed_graph.npz',pairs=pairs,origin=origin,
                logits=logits,embeddings=embeddings,
                trajectory_keys=np.asarray([t.key for t in calibration.trajectories]))
    # All source model fitting is complete before any target inference begins.
    write_json(dest/'source_models_frozen.json',dict(
        encoder_sha256=spec['encoder_sha256'],mlp_sha256=sha256(dest/'mlp/pointwise_mlp.pt'),
        calibrator_sha256=sha256(dest/'calibration/calibrator.pkl'),
        support_sha256=sha256(dest/'calibration/support.json'),
        training_seed=cfg['seed'],calibration_model_seed=cfg['calibration_seed'],
        calibration_source=manifest['roles']['calibration'],
        validation_labels_used_for_calibrator_fit=False,
        target_identity_labels_used=False,uses_M=False,uses_K=False))
    del prepared,embeddings,graph,logits,pairs,origin,stats,checkpoint,mlp_checkpoint
    gc.collect()
    pred,report = infer(roles['test'].trajectories,encoder,scorer,support,calibrator,
                        cfg,kp,kc,device,dest/'test')
    # Target labels are used for the first time in a model-related computation here,
    # AFTER infer has saved partition.npz. No test metric affects parameters.
    metrics = evaluate_partition(roles['test'].labels,pred)
    with np.load(dest/'test/candidates.npz') as cache:
        candidate_audit = evaluate_candidates_after_prediction(roles['test'].labels,cache['pairs'])
    write_json(dest/'test/metrics.json',metrics)
    write_json(dest/'test/candidate_audit.json',candidate_audit)
    result = dict(city=city,training_seed=cfg['seed'],preset_kp=kp,preset_kc=kc,metrics=metrics,
        protocol=protocol['name'],role_manifest=manifest['roles'],calibration_graph=graph_info,
        validation_labels_used_for_calibrator_fit=False,
        inference_seconds=report['inference_seconds'],candidate_pairs=report['candidate_pairs'],
        possible_pairs=report['possible_pairs'],pair_reduction_fraction=report['pair_reduction_fraction'],
        candidate_audit=candidate_audit,
        training_resources=training_resources.as_dict(),calibration_resources=calibration_resources.as_dict(),
        inference_resources=report['resources'],device_memory=report['device_memory'],
        main_wall_seconds=perf_counter()-begin,
        main_wall_scope='loading + fresh MLP training + calibration + target inference + writes + final evaluation; excludes separate offline preparation',
        uses_M=False,uses_K=False,additional_scored_pairs=0,prior_mlp_run_dependency=False)
    write_json(dest/'result.json',result)
    log('city_complete',city=city,training_seed=cfg['seed'],G_Acc=metrics['Grouping_ACC'],Pairwise_F1=metrics['Pairwise_F1'],
        clusters=metrics['n_pred_clusters'],inference_seconds=report['inference_seconds'])
    return result
