import torch

from nn_models.resnet import multi_resnet18_kd
from nn_models.resnet_tiny import multi_resnet18_tiny
from nn_models.vggnet import make_VGG 
from nn_models.vggnet import make_MobileNetV2
from model_avg import model_avg

class Server:
    def __init__(self, args):
        self.model = []
        self.param_sum = dict()
        self.param_count = dict()
        self.args = args
        # self.h = dict() ## per each branch
    def update_node_info(self, weight):
        for k in weight.keys():
            if self.param_sum[k] is None:
                self.param_sum[k] = weight[k]
            else:
                self.param_sum[k] += weight[k]
            self.param_count[k] += 1

    def avg_parameters(self):
        origin = self.model.state_dict()
        avg_parameters = model_avg(self.param_sum, self.param_count, self.args, origin)
        state_dict = {k: avg_parameters[k] for k in self.model.state_dict().keys()}
        self.model.load_state_dict(state_dict)
        
        for k in self.model.state_dict().keys():
            self.param_sum[k] = None
            self.param_count[k] = 0

    def set_initial_model(self):
        
        if self.args.dataset == 'cifar10':
            num_classes = 10
            self.model = make_VGG(norm=self.args.norm)
            # self.model.load_state_dict(torch.load('model.pt'))
        elif self.args.dataset == 'cifar100':
            num_classes = 100
            self.model = make_MobileNetV2()
            

        self.param_sum = {k: None for k in self.model.state_dict().keys()}
        self.param_count = {k: 0 for k in self.model.state_dict().keys()}
    
        state_dict = {k: self.model.state_dict()[k] for k in self.model.state_dict().keys()}
       
        self.model.load_state_dict(state_dict)


    def get_model(self):
        return self.model





