import cv2
import os
from detectron2.config import get_cfg
cfg.MODEL.WEIGHTS = "dir/to/some/checkpointfile.pth"
predictor = DefaultPredictor(cfg) #this is the model that gives predictions

input_img = cv2.imread("./input.jpg")
prediction = predictor(input_img)['instances']
#prediction is a torch.tensor-object with a number of fields. See more at https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
#for example, the number of objects found is
len(prediction)
#one may retrieve predicted bounding boxes in numpy matrix
#notice .to('cpu') must be called for cpu before converting to numpy.
img_np = prediction.pred_boxes.to('cpu').numpy()


class TI_Predictor_Base():
    def __init__(self,wd = None,cfg_dir=None,chk_dir=None,predictor_cls = DefaultPredictor,**kwargs_to_predictor):
        if wd is None:
            wd = os.getcwd()
        self.wd = wd
        if cfg_dir is None:
            cfg_dir = self.wd
        self.cfg = get_cfg().merge_from_file(cfg_dir)
        if chk_dir is None:
            chk_dir = self.wd
        self.cfg.MODEL.WEIGHTS = chk_dir
        self.predictor = predictor_cls(self.cfg,**kwargs_to_predictor)


    def __call__(self, inputs):
        pred_outputs = self.predictor(inputs)
        return self.post_process(pred_outputs)

    def post_process(self,pred_outputs):
        raise NotImplementedError

