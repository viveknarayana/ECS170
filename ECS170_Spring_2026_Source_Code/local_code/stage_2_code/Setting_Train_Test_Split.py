'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from local_code.stage_2_code.Evaluate_Multiclass_Metrics import Evaluate_Multiclass_Metrics


class Setting_Train_Test_Split(setting):
    train_file_name = 'train.csv'
    test_file_name = 'test.csv'
    
    def load_run_save_evaluate(self):
        # Stage 2 uses fixed files for train/test, no random split or CV.
        loaded_train_data = self.dataset.load_file(self.train_file_name)
        loaded_test_data = self.dataset.load_file(self.test_file_name)

        # run MethodModule
        self.method.data = {
            'train': {'X': loaded_train_data['X'], 'y': loaded_train_data['y']},
            'test': {'X': loaded_test_data['X'], 'y': loaded_test_data['y']}
        }
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()
            
        self.evaluate.data = learned_result
        acc = self.evaluate.evaluate()

        multiclass = Evaluate_Multiclass_Metrics('multiclass P/R/F1', '')
        multiclass.data = learned_result
        multiclass.evaluate()

        return acc, None

        
