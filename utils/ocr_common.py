from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class OCRCommon:
    def __init__(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        # config['weights'] = './weights/transformerocr.pth'
        config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
        config['cnn']['pretrained'] = False
        config['device'] = 'cuda:0'
        config['predictor']['beamsearch'] = False

        self.detector = Predictor(config)

    def predict(self, img):
        img = Image.fromarray(img)

        return self.detector.predict(img)
