"""Identity-level source holdout and provenance shared by all three seeds."""
import copy

from .data import load_splits, split_source_users, describe_split
from .io import write_json, log

PROTOCOL_NAME = 'source_user_calibration_90_10_v1'
EXPECTED_SEEDS = [42, 43, 44]


def validate_protocol(protocol):
    expected = dict(name=PROTOCOL_NAME, split_seed=42, calibration_split_seed=42,
        calibration_numerator=1, calibration_denominator=10,
        calibration_count_rounding='nearest_half_up', validation_sampling_seed=42,
        calibration_model_seed=42, budget_rounding=100)
    if protocol != expected:
        raise ValueError('This release requires the fixed 90/10 protocol and split seed 42')


def training_config(config, seed):
    validate_protocol(config['protocol'])
    if config['training_seeds'] != EXPECTED_SEEDS or type(seed) is not int or seed not in EXPECTED_SEEDS:
        raise ValueError('Paired Transformer/MLP training seeds must be 42, 43, 44')
    c = copy.deepcopy(config['training'])
    if 'seed' in c or 'seed' in c['mlp']:
        raise ValueError('Use training_seeds only; separate MLP seeds are not permitted')
    c.update(seed=seed, validation_sampling_seed=config['protocol']['validation_sampling_seed'],
             calibration_seed=config['protocol']['calibration_model_seed'])
    return c


def load_protocol(path, protocol):
    validate_protocol(protocol)
    outer, original = load_splits(path, protocol['split_seed'])
    neural, calibration = split_source_users(outer['train'], protocol['calibration_split_seed'])
    roles = dict(neural_train=neural, calibration=calibration, val=outer['val'], test=outer['test'])
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if set(roles[a].users) & set(roles[b].users):
                raise AssertionError(f'Identity overlap between {a} and {b}')
    manifest = dict(protocol=copy.deepcopy(protocol), original=original,
                    roles={name: describe_split(s) for name, s in roles.items()},
                    all_four_roles_identity_disjoint=True,
                    source_identity_union_preserved=True)
    log('source_roles_fixed', **manifest['roles'])
    return roles, manifest


def save_protocol(dest, roles, manifest):
    write_json(dest/'data_manifest.json', manifest)
    # UUIDs are audit/supervision metadata. Feature extraction never reads this file.
    write_json(dest/'identity_allocation.json', {name: s.users for name, s in roles.items()})


def neural_provenance(train, validation, cfg):
    if set(train.users) & set(validation.users):
        raise ValueError('Neural training and validation identities overlap')
    return dict(protocol=PROTOCOL_NAME, training_role='neural_train_90_percent_users',
                seed=cfg['seed'], neural_train=describe_split(train), validation=describe_split(validation),
                validation_sampling_seed=cfg['validation_sampling_seed'],
                calibration_users_in_neural_training=False)


def validate_neural_checkpoint(checkpoint, manifest, cfg):
    actual = checkpoint.get('provenance', {})
    expected = dict(protocol=PROTOCOL_NAME, training_role='neural_train_90_percent_users',
        seed=cfg['seed'], neural_train=manifest['roles']['neural_train'],
        validation=manifest['roles']['val'], validation_sampling_seed=cfg['validation_sampling_seed'],
        calibration_users_in_neural_training=False)
    if actual != expected or checkpoint.get('seed') != cfg['seed']:
        raise ValueError('Checkpoint seed/user provenance differs: old all-source models cannot be reused')
