'''
Setting for stage 4 joke generation: train, then print sample generations.
'''

import random
from local_code.base_class.setting import setting
from local_code.stage_4_code.Dataset_Loader_Generation import (
    BOS_ID, EOS_ID, PAD_ID, UNK_ID,
)


class Setting_Train_Generate(setting):

    # script can override these; defaults give a reasonable run with no extra config
    num_dataset_seeds = 3
    freestyle_seeds = None  # e.g. [['my', 'dog', 'said'], ['the', 'computer', 'is']]

    def _seeds_from_dataset(self, encoded_jokes, id_to_token, n):
        # take the first three real (non-special) tokens of n random jokes
        rng = random.Random(2)
        picks = rng.sample(range(len(encoded_jokes)), min(n, len(encoded_jokes)))
        seeds = []
        for idx in picks:
            words = []
            for tid in encoded_jokes[idx]:
                if tid in (BOS_ID, EOS_ID, PAD_ID, UNK_ID):
                    continue
                words.append(id_to_token[tid])
                if len(words) == 3:
                    break
            if len(words) == 3:
                seeds.append(words)
        return seeds

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load_file(self.dataset.dataset_source_file_name)

        self.method.vocab_size = len(self.dataset.vocab)
        self.method.vocab = self.dataset.vocab
        self.method.id_to_token = self.dataset.id_to_token

        self.method.data = {
            'train': {'X': loaded_data['train']['X'], 'y': loaded_data['train']['y']},
            'test':  {'X': loaded_data['test']['X'],  'y': loaded_data['test']['y']}
        }
        learned_result = self.method.run()

        dataset_seeds = self._seeds_from_dataset(
            loaded_data['train']['X'], self.dataset.id_to_token, self.num_dataset_seeds
        )

        print('sample generations (seeds drawn from the dataset):')
        from_data_outputs = []
        for seed in dataset_seeds:
            text = self.method.generate(seed)
            from_data_outputs.append({'seed': seed, 'text': text})
            print(f'  {" ".join(seed)} | {text}')

        freestyle_outputs = []
        if self.freestyle_seeds:
            print('sample generations (freestyle seeds):')
            for seed in self.freestyle_seeds:
                text = self.method.generate(seed)
                freestyle_outputs.append({'seed': seed, 'text': text})
                print(f'  {" ".join(seed)} | {text}')

        learned_result['generations_from_data'] = from_data_outputs
        learned_result['generations_freestyle'] = freestyle_outputs

        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()

        return None, None
