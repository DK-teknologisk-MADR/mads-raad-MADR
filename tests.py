import unittest
from pruners import SHA

from D2TIDefaults import get_data_dicts, get_file_pairs , register_data

data_dir = "/pers_files/Filet/annotations/Combined/1024x1024"  # TODO:Change test_data
splits = ['train', 'val']
class Tester(unittest.TestCase):

    def test_get_data_dicts(self):
        for split in splits:
            data = get_data_dicts(data_dir, split)
            #TODO: Assert stuff

    def test_register_data(self):
        COCO_dicts = {split: get_data_dicts(data_dir,split) for split in splits }
        register_data('filet',['train','val'],COCO_dicts)

    def test_pruner_SHA(self):
        inputs= [18,7,14,1,5,2,9,3,5,7,1,8,11,2,3,4,   2,1,7,3,     1]
        input_dict0 = {i : inputs[i] for i in range(16)}
        winner0 = [0,2,12,7]
        winner1 = [12]
        input_dict1 = { winner0[i]: inputs[i+16] for i in range(4)}
        input_dict = input_dict0
        loop_id = 0
        done = False
        trial_id = 0
        sha=SHA(21,4)
        prune_count = 0
        expected_prune = [set(range(20)) - set(winner0),set(winner1)-set(winner0)]
        while not done:
            self.assertTrue(loop_id<21,f"sha should have completed by trial 21 with factor 4, and 16 participants, but loop runs at iter {loop_id}")
            trial_id ,pruned, done = sha.report_and_get_next_trial(input_dict[trial_id])
            if pruned:
                input_dict = input_dict1
                self.assertTrue(set(pruned)==expected_prune, f"pruned {pruned}, but expected {expected_prune[prune_count]} at trial {trial_id}")
                prune_count+=1
            loop_id +=1

if __name__ == '__main__':
    unittest.main()