"""
Runs all 6 stage-4 variants sequentially.
Generation first (fast), then classification (slow).
Each variant's full stdout/stderr goes to <result_dir>/run.log.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
RESULT_DIR = PROJECT_ROOT / 'result' / 'stage_4_result'


RUNS = [
    ('generation rnn',      'script_rnn_generation.py',     'rnn',  'GEN_rnn'),
    ('generation lstm',     'script_rnn_generation.py',     'lstm', 'GEN_lstm'),
    ('generation gru',      'script_rnn_generation.py',     'gru',  'GEN_gru'),
    ('classification rnn',  'script_rnn_classification.py', 'rnn',  'RNN_IMDB_rnn'),
    ('classification lstm', 'script_rnn_classification.py', 'lstm', 'RNN_IMDB_lstm'),
    ('classification gru',  'script_rnn_classification.py', 'gru',  'RNN_IMDB_gru'),
]


def now():
    return datetime.now().strftime('%H:%M:%S')


def main():
    start = time.time()
    print(f'starting all 6 runs at {datetime.now()}')

    for label, script, variant, subdir in RUNS:
        out_dir = RESULT_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / 'run.log'

        print(f'\n[{now()}] >>> {label}')
        run_start = time.time()
        with open(log_path, 'w') as f:
            result = subprocess.run(
                [sys.executable, script, '--variant', variant],
                cwd=THIS_DIR,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.time() - run_start
        status = 'ok' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
        print(f'[{now()}] <<< {label} {status} in {elapsed:.0f}s. log: {log_path}')

    total = time.time() - start
    print(f'\nall runs finished at {datetime.now()} (elapsed: {total:.0f}s)')


if __name__ == '__main__':
    main()
