import sys
from pathlib import Path

### NEED TO RUN OTHER VARIANTS OF THIS ALL FIRST DATASET STUFF


# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Method_RNN_LSTM import Method_RNN_LSTM
from local_code.stage_4_code.Method_RNN_GRU import Method_RNN_GRU
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os
import argparse


"""
Stage 4 RNN classification runner (IMDb).
Usage examples:
  python script_rnn_classification.py --variant rnn
  python script_rnn_classification.py --variant lstm
  python script_rnn_classification.py --variant gru
  python script_rnn_classification.py --variant all
"""


METHOD_BY_VARIANT = {
    'rnn':  (Method_RNN,'RNN IMDB'),
    'lstm': (Method_RNN_LSTM,'RNN IMDB LSTM'),
    'gru':  (Method_RNN_GRU,'RNN IMDB GRU'),
}


def run_single_variant(variant):
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('stage 4 IMDb', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/'
    data_obj.dataset_source_file_name = ''

    method_cls, method_name = METHOD_BY_VARIANT[variant]
    method_obj = method_cls(method_name, '')

    result_obj = Result_Saver('saver', '')
    run_tag = f'IMDB_{variant}'
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/RNN_{run_tag}/'
    result_obj.result_destination_file_name = f'prediction_result_{run_tag}'
    os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    print(f'Start (IMDb {variant})')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print(f'Overall Performance (IMDb {variant})')
    print(f'RNN Accuracy: {mean_score} +/- {std_score}')
    print(f'Finish (IMDb {variant})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--variant',
        type=str,
        default='rnn',
        choices=['rnn', 'lstm', 'gru', 'all'],
        help='Recurrent cell variant: rnn (baseline), lstm, gru, or all'
    )
    args = parser.parse_args()

    variants = ['rnn', 'lstm', 'gru'] if args.variant == 'all' else [args.variant]
    for v in variants:
        run_single_variant(v)
