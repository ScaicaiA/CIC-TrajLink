import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cic.io import read_json
from cic.runtime import commit, committed, preserve_incomplete, run_lock
from prepare_data import prepare

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_core_modules_and_pipeline_are_identical_to_main_source(self):
        reference = read_json(ROOT / 'reference/source_integrity.json')
        for name, expected in reference['files'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)
        tree = ast.parse((ROOT / 'main.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_city')
        self.assertEqual(hashlib.sha256(ast.dump(run, include_attributes=False).encode()).hexdigest(),
                         reference['main_run_city_ast_sha256'])

    def test_fixed_budgets_match_every_archived_city_seed(self):
        config = read_json(ROOT / 'config/four_cities.json')
        reference = read_json(ROOT / 'reference/main_results.json')
        self.assertEqual(len(reference['per_seed']), 12)
        self.assertNotIn('selection', config)
        for row in reference['per_seed']:
            self.assertEqual(config['cities'][row['city']]['fixed_k'][str(row['seed'])],
                             dict(kp=row['kp'], kc=row['kc']))

    def test_completed_stage_integrity_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'stage'
            folder.mkdir()
            (folder / 'weights').write_bytes(b'trained weights')
            expected = dict(city='Tokyo', seed=42)
            self.assertFalse(committed(folder, expected))
            commit(folder, expected)
            self.assertTrue(committed(folder, expected))
            with self.assertRaises(ValueError):
                committed(folder, dict(city='Tokyo', seed=43))
            (folder / 'weights').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                committed(folder, expected)

    def test_incomplete_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'seed_42/main/Tokyo'
            folder.mkdir(parents=True)
            (folder / 'partial').write_bytes(b'existing output')
            preserve_incomplete(folder, root)
            self.assertFalse(folder.exists())
            saved = list((root / 'incomplete_attempts').rglob('partial'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), b'existing output')

    def test_live_run_cannot_be_started_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.run.lock'
            with run_lock(path):
                code = 'from pathlib import Path; from cic.runtime import run_lock; import sys\nwith run_lock(Path(sys.argv[1])): pass'
                result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('already using this run', result.stderr)
            with run_lock(path):
                pass

    def test_data_unpack_is_exact_and_rejects_modified_existing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b'columns\r\n"content,with commas"\r\n'
            archive = root / 'city.csv.gz'
            archive.write_bytes(gzip.compress(payload, mtime=0))
            config = root / 'config.json'
            config.write_text(json.dumps(dict(cities=dict(Tokyo=dict(data='city.csv', archive='city.csv.gz',
                data_sha256=hashlib.sha256(payload).hexdigest(), archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))))
            prepare(config, verify_only=True)
            self.assertFalse((root / 'city.csv').exists())
            prepare(config)
            self.assertEqual((root / 'city.csv').read_bytes(), payload)
            prepare(config)
            (root / 'city.csv').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare(config)
            self.assertEqual((root / 'city.csv').read_bytes(), b'changed')
