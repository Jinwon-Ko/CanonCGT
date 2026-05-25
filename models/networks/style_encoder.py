from models.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_modules import Embedding_Net


class Net(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(Net, self).__init__(cfg)

        self.Embedding_Net = Embedding_Net(cfg)
        # self.Destyler = LookUpTable_Estimator(cfg)
        # self.Restyler = LookUpTable_Estimator(cfg)

    def forward(self, img):
        return {'style_vector': self.Embedding_Net(img)}
