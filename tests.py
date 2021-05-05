import unittest

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


if __name__ == '__main__':
    unittest.main()