import sys
from pathlib import Path

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os
import argparse


"""
Stage 3 CNN runner.
Usage examples:
  python script_cnn.py --dataset MNIST
  python script_cnn.py --dataset ORL
  python script_cnn.py --dataset CIFAR
  python script_cnn.py --dataset ALL
"""


def run_single_dataset(dataset_name):
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader('stage 3', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = dataset_name

    method_obj = Method_CNN('convolutional neural network', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = f'../../result/stage_3_result/CNN_{dataset_name}/'
    result_obj.result_destination_file_name = f'prediction_result_{dataset_name}'
    os.makedirs(result_obj.result_destination_folder_path, exist_ok=True)

    setting_obj = Setting_Train_Test_Split('train test split', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start (' + dataset_name + ') ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance (' + dataset_name + ') ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))

    # ---- quick inference sanity check on one test image ---
    loaded_data = data_obj.load_file(dataset_name)
    test_index = 0
    sample_image = loaded_data['test']['X'][test_index]
    sample_true_label = loaded_data['test']['y'][test_index]
    sample_pred_label = int(method_obj.test([sample_image])[0].item())
    print(
        'Quick check (' + dataset_name + '): '
        + 'test_index=' + str(test_index)
        + ', pred=' + str(sample_pred_label)
        + ', true=' + str(sample_true_label)
    )

    print('************ Finish (' + dataset_name + ') ************')
    # ------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='MNIST',
        choices=['MNIST', 'ORL', 'CIFAR', 'ALL'],
        help='Dataset to run: MNIST, ORL, CIFAR, or ALL'
    )
    args = parser.parse_args()

    selected = args.dataset.upper()
    datasets = ['MNIST', 'ORL', 'CIFAR'] if selected == 'ALL' else [selected]
    for ds in datasets:
        run_single_dataset(ds)
