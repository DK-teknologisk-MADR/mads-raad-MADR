import unittest
from pruners import SHA
import numpy as np



class Tester(unittest.TestCase):
    data_dir = "/pers_files/Filet/annotations/Combined/1024x1024"  # TODO:Change test_data
    splits = ['train', 'val']


    def sha_routine(self,inputs,expected_prunes , expected_iter,sha):
        prune_count = 0
        loop_id = 0
        done = False
        trial_id = 0
        while not done:
            self.assertTrue(loop_id<expected_iter,f"sha should have completed by trial 21 with factor 4, and 16 participants, but loop runs at iter {loop_id}")
            trial_id ,pruned, done = sha.report_and_get_next_trial(inputs[prune_count][trial_id])
            if pruned:
                self.assertTrue(set(pruned)==expected_prunes[prune_count], f"pruned {pruned}, but expected {expected_prunes[prune_count]} at trial {trial_id}")
                prune_count+=1
            loop_id +=1

    def test_pruner_SHA(self):
        inputs = np.zeros(shape=(3,16))
        inputs[0] = np.array([1,3,7,1,2,-9,20,2,11,8,3,7,5,3,9,-4])
        inputs[1] = np.array([5,1.9,14,7,9,1,4,2,11,15,11,0,4,11,7.4,11])
        inputs[2] = np.array([18,-0.4,14,7,9,-4,15,2,11,2,11,1,4,11,3.5,11])

        sha=SHA(32,4)
        self.assertEqual(sha.participants, 16, f'got instead{(sha.participants)}')
        self.assertEqual(sha.rungs,3)
        expected_prunes = [set(range(16))-{6,8,9,14},{6,8,9,14}-{9}]
        self.sha_routine(inputs,expected_prunes,expected_iter=21,sha=sha)


        sha = SHA(max_res= 31, factor= 3,topK=3)
        self.assertEqual(sha.participants, 9)
        self.assertEqual(sha.rungs, 3)
        self.assertEqual(sha.rungs_to_skip, 2)

        expected_prunes = [set(range(9)) - {6,8,2}]
        self.sha_routine(inputs,expected_prunes,expected_iter=9,sha=sha)

if __name__ == '__main__':
    unittest.main()
