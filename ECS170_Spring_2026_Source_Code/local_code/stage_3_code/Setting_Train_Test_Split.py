'''
Concrete SettingModule class for stage 3 train/test setting.
'''

from local_code.base_class.setting import setting
from local_code.stage_3_code.Evaluate_Multiclass_Metrics import Evaluate_Multiclass_Metrics


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load_file(self.dataset.dataset_source_file_name)

        self.method.data = {
            'train': {'X': loaded_data['train']['X'], 'y': loaded_data['train']['y']},
            'test': {'X': loaded_data['test']['X'], 'y': loaded_data['test']['y']}
        }
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()

        self.evaluate.data = learned_result
        acc = self.evaluate.evaluate()

        multiclass = Evaluate_Multiclass_Metrics('multiclass P/R/F1', '')
        multiclass.data = learned_result
        multiclass.evaluate()

        return acc, None
