import D2TIDefaults as D2TI
from pruners import SHA

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog
class ex_hyperopt(D2_hyperopt):
    def __init__(kwargs_to_super):
        super().__init__(**kwargs)

    def initialize(self):
        for i in range(self.pruner.participants):
            suggested_cfg,params = self.suggest_cfg(i)
            to_json = [suggested_cfg,params]
            os.makedirs(suggested_cfg.OUTPUT_DIR, exist_ok=True)
            with open(f'{suggested_cfg.OUTPUT_DIR}/params.json', 'w') as fp:
              json.dump(to_json, fp)

    def suggest_values(self,typ_params):
        params_wo_name = deepcopy(params)
        params_wo_name.pop('name')
        if typ == "int":
            return randint(**params_wo_name)
        elif typ == "float":
            return uniform(**params_wo_name)

    def before_pruned(self):
        print('and thou said to thee, do prune thy ', pruned_ids)


splits = ['train','val']

#all datasets must be registered
for d in ["train", "val"]:
    DatasetCatalog.register("filet_" + d, lambda d=d: get_data_dicts(d))
    MetadataCatalog.get("filet_" + d).set(thing_classes=["filet"])
filet_metadata = MetadataCatalog.get('filet_train')

#below we initialize a cfg
def initialize_cfg(model_name="",cfg=None):
  if cfg is None:
      cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(f'{model_name}.yaml'))
  cfg.DATASETS.TRAIN = ("filet_train",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2 #SHOULD BE SET HIGHER BECAUSE WE HAVE NICE CPUS AT TI
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{model_name}.yaml')
  cfg.SOLVER.IMS_PER_BATCH = 3
  cfg.OUTPUT_DIR = f'./{model_name}_output'
  cfg.SOLVER.BASE_LR = 0.00025
  cfg.SOLVER.MAX_ITER = 200
#  cfg.SOLVER.STEPS = (20,)
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  #(default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # which is filet
  return cfg




#example input

model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
solver_dict = {
    'BASE_LR' : ('float',{'name' : 'lr', 'low' : 0.0001 , 'high' : 0.001}),
    'GAMMA'  : ('float', {'name' : 'lr', 'low' : 0.005 , 'high' : 0.8}),
    'STEPS' : ('int_list', {'name' : 'lr', 'low' :2000, 'high' :25000  }),
    }
backbone_dict = {
    'FREEZE_AT' : ('int',{'name' : 'freeze', 'low' : 0 , 'high' : 2})
    }
model_dict = {'BACKBONE' : backbone_dict}

hp_dict = {'SOLVER' : solver_dict,
           'MODEL' : model_dict,
          }
model_dicts = []
for model_name_ in model_names:
    model_dict = {'name' : model_name_,
                'base_cfg': initialize_cfg(model_name_),
                'hp_dict' : {'SOLVER' : solver_dict}}




cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold


task = "bbox"
val_data_name ="filet_val"
evaluator = COCOEvaluator(val_data_name, [task], False,output_dir="/test/output/")
hyperopt = D2_hyperopt(model_dict,task,evaluator,output_dir="test/output/",super_step_size=100,max_epochs=250000,pruner_cls=SHA)
hyperopt.start()
