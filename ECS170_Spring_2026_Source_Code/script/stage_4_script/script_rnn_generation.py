import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader_Generation
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from local_code.stage_4_code.Method_RNN_Generator_LSTM import Method_RNN_Generator_LSTM
from local_code.stage_4_code.Method_RNN_Generator_GRU import Method_RNN_Generator_GRU
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Generate import Setting_Train_Generate
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os
import argparse


"""
Stage 4 RNN joke generation runner.
Usage examples:
  python script_rnn_generation.py --variant rnn
  python script_rnn_generation.py --variant lstm
  python script_rnn_generation.py --variant gru
  python script_rnn_generation.py --variant all
"""


METHOD_BY_VARIANT = {
    'rnn':  (Method_RNN_Generator,      'GEN RNN'),
    'lstm': (Method_RNN_Generator_LSTM, 'GEN LSTM'),
    'gru':  (Method_RNN_Generator_GRU,  'GEN GRU'),
}

# made-up seeds for the "doesn't necessarily exist in the dataset" part of 4-4
FREESTYLE_SEEDS = [
    ['my', 'dog', 'said'],
    ['the', 'computer', 'is'],
    ['a', 'horse', 'walks'],
]


def run_single_variant(variant):
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader_Generation('stage 4 jokes', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = ''

    method_cls, method_name = METHOD_BY_VARIANT[variant]
    method_obj = method_cls(method_name, '')

    result_obj = Result_Saver('saver', '')
    run_tag = f'GEN_{variant}'
    result_obj.result_destination_folder_path = f'../../result/stage_4_result/{run_tag}/'
    result_obj.result_destination_file_name = f'generation_result_{variant}'
    os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)

    setting_obj = Setting_Train_Generate('train and generate', '')
    setting_obj.freestyle_seeds = FREESTYLE_SEEDS
    evaluate_obj = Evaluate_Accuracy('unused', '')

    print(f'Start (jokes {variant})')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print(f'Finish (jokes {variant})')


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
