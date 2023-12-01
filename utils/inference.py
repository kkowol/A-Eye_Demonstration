import numpy as np
import torch
import os

from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
from models.fast_scnn import FastSCNN
from models.bisenetv2 import BiSeNetV2
from models.ebunet import EBUNet
from torch2trt import TRTModule
from utils.carla_dataloader import Carla
import config as cfg
import carla

def carla_colorize(arr):
    imc = arr.convert('P')
    imc.putpalette(Carla.color_palette_train_ids)
    return imc.convert('RGB')


def img_enlargement(img, width, height):
    """
    enlarge the input data with factor 3
    comparision of the available filters:
    https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
    """
    return img.resize((width*3, height*3))


def output_folders_inference():
    """
    create output folders for inference script with counting the existing folders
    """
    path = os.path.join(os.getcwd(), 'output')
    nr_files = len(os.listdir(path))
    path = os.path.join(path, '{}_%04i'.format(cfg.name_out_folder)) % nr_files
    os.mkdir(path)
    return path


class Inference():
    def __init__(self, ckpt, weather):
        if ckpt:
            self.ckpt = ckpt
            self.weather = weather
            self.load_mean_std()

            models =['FastSCNN', 'bisenetv2', 'EBUNet']
            for model in models:
                if model in ckpt:
                    self.model_name = model

            if self.model_name == models[0]:
                self.network = FastSCNN(in_channels=3, num_classes=Carla.num_train_ids)
                # self.network = TRTModule()
            elif self.model_name == models[1]:
                self.network = BiSeNetV2(n_classes=Carla.num_train_ids)
                self.network = TRTModule()
            elif self.model_name == models[2]:
                self.network = EBUNet(classes=Carla.num_train_ids)
                self.network = TRTModule()
            self.network.load_state_dict(torch.load(os.path.join(os.getcwd(), 'weights', self.ckpt + '.pth')))
            self.network.cuda().eval()

    def processing(self, image):
        """
        inference function
        """
        #----------- preprocessing -----------
        image.convert(carla.ColorConverter.Raw)     # raw data needed!!!
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] # swap first and third channel (bgr-->rgb)
        img = array.copy()
        #----------- inference -----------
        x = self.torch_transform(img).unsqueeze_(0).cuda() # unsqueeze --> add a channel
        # with torch.no_grad():
        if self.model_name == 'bisenetv2':
            y_trt = self.network(x)[0]
        else:
            y_trt = self.network(x)
        pred = y_trt.argmax(dim=1)[0].cpu()#.numpy()
        pred = ToPILImage()(pred.to(dtype=torch.uint8))
        #----------- unique/converting -----------
        mask = np.array(carla_colorize(pred)) 
        return mask
    
    def load_mean_std(self):
        if self.weather == 'mix':
            mean   = (0.4573, 0.4412, 0.4223)
            std    = (0.1719, 0.1657, 0.1574)
        elif self.weather == 'clear' or self.weather == None:
            mean   = (0.5359, 0.5150, 0.4899)
            std    = (0.2362, 0.2269, 0.2197)
        elif self.weather == 'rain':
            mean   = (0.5373, 0.5167, 0.4927)
            std    = (0.2094, 0.2027, 0.2008)
        elif self.weather == 'fog':
            mean   = (0.6158, 0.6049, 0.5973)
            std    = (0.1795, 0.1745, 0.1702)
        elif self.weather == 'night':
            mean   = (0.3195, 0.2995, 0.2713)
            std    = (0.1844, 0.1750, 0.1579)
        self.torch_transform = Compose([ToTensor(), Normalize(mean, std)])