import os
import torch
import numpy as np
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if(torch.backends.mps.is_available()):
                device = torch.device('mps')
                print('Use GPU: mps')
            elif self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                if self.args.use_multi_gpu:
                    device = torch.device('cuda:{}'.format(self.args.gpu))
                    print('Use GPU: cuda:{}'.format(self.args.devices))
                else:
                    device = torch.device('cuda:{}'.format(self.args.gpu))
                    print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')

        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
