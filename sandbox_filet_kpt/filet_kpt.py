import numpy as np
from predictors import TI_Predictor_Base
import cv2 as cv
import numbers
from matplotlib import pyplot as plt
import time

'''
#fake_dt construction
a=700
b=200
n=4
xs = np.random.uniform(-a,a,100)
y_sgns = np.repeat([1,-1],50)
def ellipse_y_from_x(x,y_sgn,a,b,n=2,rot_rad=0):
    y = y_sgn*( b**n*(1-(x/a)**n) )**(1/n)
    return y
ys = ellipse_y_from_x(xs,y_sgns,a,b,n) -100-100/(600**2)*xs**2
cont_bef_sort = np.array(list(zip(xs,ys)))

xsp = np.argsort(xs[:50])
xsm = np.argsort(xs[50:])[::-1]+50
xs_ind = np.concatenate([xsp,xsm])
xs_ind = np.concatenate([xs_ind[17:],xs_ind[:17]])
cont_np = cont_bef_sort[xs_ind,:]
plt.scatter(cont_np[:,0],cont_np[:,1],c=list(range(100)))
'''


class Filet_KPT_Predictor(TI_Predictor_Base):
    def __init__(self,cfg_fp,chk_fp,n_split=21,thresh_up=40,thresh_below=10):
        super().__init__(cfg_fp=cfg_fp,chk_fp=chk_fp)
        self.thresh_below = thresh_below
        self.thresh_up = thresh_up
        self.n_split = n_split

    def interpolate_if_needed(self,indices,to_add,x1,x2,x2_ind):
        taxi_norm = np.sum(np.abs(x2 - x1))
        if taxi_norm > self.thresh_up:
            nr_points = np.int(taxi_norm / self.thresh_below)
            to_add.extend(list(np.linspace(x2, x1, nr_points)))
            indices.extend([x2_ind for _ in range(nr_points)])


    def post_process(self,pred_outputs):
        pred_masks = pred_outputs['instances'].pred_masks.to('cpu').numpy()[0]
        pred_masks = (pred_masks*1).astype('uint8')
        contours = cv.findContours(pred_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cont_np = np.array([x.flatten() for x in contours[0]])
        boundaries =np.array([ np.min(cont_np), np.max(cont_np) ])
        filet_length = boundaries[1] - boundaries[0]

        to_add = []
        indices = []
        i = 1
        # here we mark what to add
        #plt.scatter(cont_np[:,0], cont_np[:,1], c=list(range(cont_np.shape[0])))
        while (i < cont_np.shape[0]):
            self.interpolate_if_needed(indices,to_add,cont_np[i-1],cont_np[i],i)
            i += 1
        last_ind = len(cont_np) - 1
        self.interpolate_if_needed(indices,to_add,cont_np[last_ind], cont_np[0],last_ind)
        if indices:
          z = np.insert(cont_np, indices, to_add, axis=0)
        else:
          z=cont_np
        ords = z[:, 0].argsort()
        z = z[ords, :]
        split_pts = np.linspace(z[0, 0], z[-1, 0], self.n_split + 1)
        start_window_time =time.time()
        window_start_index = 0
        mean_vals = []
        for window_end_x in split_pts[1:]:
            window_end_index = np.where(z[:, 0] < window_end_x)[0][-1]
            y_vals = z[window_start_index:window_end_index + 1, 1]
            y_bar = np.mean(y_vals)
            mean_over = np.mean(y_vals[y_vals > y_bar])
            mean_under = np.mean(y_vals[y_vals < y_bar])
            mean_vals.append((mean_over + mean_under) / 2)
            window_start_index = window_end_index
        x_vals = split_pts[1:] - 1 / (2 * (split_pts[1:]-split_pts[:-1]))
        #plt.scatter(z[:, 0], z[:, 1], c=list(range(z.shape[0])))
        #plt.scatter(x_vals, mean_vals)
        return np.transpose(np.array([x_vals,mean_vals]))


predictor =Filet_KPT_Predictor('filet_cfg.yaml','model_final.pkl',n_split=21)
filet_img = cv.imread('filet_faux.jpg')
start_time = time.time()
pred_output = predictor(filet_img)