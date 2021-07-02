import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt

def weight2img(opt, target_weight: torch.tensor):
    weight = target_weight.detach().cpu().numpy()
    wMax, wMin = weight.max(), weight.min()
    weight = (weight-wMin) / (wMax-wMin+1e-30)
    
    HM = np.zeros((weight.shape[2]*weight.shape[1], weight.shape[3]*weight.shape[0]))
    for x in range(weight.shape[0]):
        for y in range(weight.shape[1]):
            HM[y*weight.shape[2]:(y+1)*weight.shape[2],x*weight.shape[3]:(x+1)*weight.shape[3]] = weight[x][y]

    image = Image.fromarray(np.uint8(cm.bwr(HM)*255)) #plasma
    
    if opt.get_gif:
        return image
    
    if opt.draw_pic:    
        image.show()