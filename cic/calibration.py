"""The retained fourteen evidence features and fresh source calibration."""
from pathlib import Path
from time import perf_counter
import pickle
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning

from .edge_calibration import edge_features, calibrated_log_odds
from .neighborhood import neighbor_agreement
from .poi_evidence import encode_nodes, evidence_on_pairs
from .io import write_json, log
from .data import describe_split

FEATURE_NAMES = ['raw_mlp_logit','mlp_endpoint_z_min','mlp_endpoint_z_max',
    'encoder_cosine','encoder_endpoint_z_min','encoder_endpoint_z_max',
    'common_positive_support','poi_idf_cosine','poi_energy_coverage_min',
    'poi_energy_coverage_max','log_visit_count_min','log_visit_count_max',
    'log_unique_poi_min','log_unique_poi_max']


def fit_support(x, y, seed):
    if len(x) != len(y) or set(np.unique(y)) != {0,1}:
        raise ValueError('Calibration graph needs observed same-user and different-user edges')
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,
                               class_weight=None,random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(scaler.transform(x),y)
    coef = model.coef_[0]/scaler.scale_
    intercept = float(model.intercept_[0]-coef@scaler.mean_)
    return dict(feature_indices=list(range(6)),raw_coefficients=coef.tolist(),raw_intercept=intercept,
                feature_names=FEATURE_NAMES[:6],fit_edges=len(y),fit_positive_edges=int(y.sum()),
                class_weight=None,negative_sampling=False,seed=seed)


def assemble(graph, trajectories, support, support_floor, base=None):
    begin = perf_counter()
    if base is None:
        base = edge_features(graph['n'],graph['pairs'],graph['logits'],graph['embeddings'])
    u = calibrated_log_odds(base,support)
    h, common = neighbor_agreement(graph['n'],graph['pairs'],u,logit_floor=support_floor)
    extra,poi = evidence_on_pairs(encode_nodes(trajectories),graph['pairs'])
    x = np.column_stack((base,h,extra))
    if x.shape != (len(graph['pairs']),14) or not np.all(np.isfinite(x)):
        raise AssertionError('Invalid fourteen-feature matrix')
    return x,dict(feature_names=FEATURE_NAMES,common_neighbors=common,poi=poi,
                  seconds=perf_counter()-begin,additional_pairs=0)


def fit_calibration(graph, source_calibration, cfg, dest):
    """Fit LR and HGB on the independent 10% source-user graph only.

    The caller supplies a complete standalone calibration cohort, never an
    induced subgraph of validation. No validation/test object is accepted.
    """
    trajectories, labels = source_calibration.trajectories, source_calibration.labels
    if graph['n'] != len(trajectories) or len(labels) != graph['n']:
        raise ValueError('Calibration graph and independent source cohort disagree')
    dest = Path(dest)
    dest.mkdir(parents=True,exist_ok=False)
    y = (labels[graph['pairs'][:,0]] == labels[graph['pairs'][:,1]]).astype(np.int64)
    base = edge_features(graph['n'],graph['pairs'],graph['logits'],graph['embeddings'])
    support = fit_support(base,y,cfg['calibration_seed'])
    x,info = assemble(graph,trajectories,support,cfg['support_logit_floor'],base)
    log('fit_edge_calibrator',observed_edges=len(y),positives=int(y.sum()),features=14)
    model = HistGradientBoostingClassifier(**cfg['calibrator'],random_state=cfg['calibration_seed']).fit(x,y)
    with (dest/'calibrator.pkl').open('wb') as f:
        pickle.dump(model,f)
    write_json(dest/'support.json',support)
    write_json(dest/'training.json',dict(fit_edges=len(y),positive_edges=int(y.sum()),
        class_rebalancing=False,features=info,parameters=cfg['calibrator'],
        fixed_support_logit_floor=cfg['support_logit_floor'],
        labels_scope='independent 10 percent of ORIGINAL TRAINING identities only',
        calibration_model_seed=cfg['calibration_seed'],
        source_cohort=describe_split(source_calibration),
        validation_labels_used_for_fitting=False, test_labels_used_for_fitting=False,
        observed_positive_fraction=float(y.mean())))
    return support,model


def calibrated_probabilities(graph, trajectories, support, model, cfg):
    if len(graph['pairs']) == 0:
        return np.empty(0),dict(feature_names=FEATURE_NAMES,additional_pairs=0)
    x,info = assemble(graph,trajectories,support,cfg['support_logit_floor'])
    if model.n_features_in_ != 14:
        raise ValueError('This main pipeline requires all fourteen features')
    p = model.predict_proba(x)[:,1]
    return p,info
