import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import tensorflow as tf
import gc
from BENDR.dn3_ext import LinearHeadBENDR
import BENDR.dn3_ext as dn3_ext

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWrapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None

    @abstractmethod
    def get_cutted_model(self, bottleneck):
        pass

    def get_gradient(self, acts, y, bottleneck_name):
        inputs = torch.autograd.Variable(torch.tensor(np.array(acts)).to(self.model.device), requires_grad=True)
        #targets = (y[0] * torch.ones(inputs.size(0))).long().to(device)

        cutted_model = self.get_cutted_model(bottleneck_name).to(self.model.device)
        cutted_model.eval()
        outputs = cutted_model(inputs)

        grads = -torch.autograd.grad(outputs[:, y[0]], inputs)[0]
        
        grads = grads.detach().cpu().numpy()

        cutted_model = None
        
        del inputs
        del outputs
        gc.collect()
        torch.cuda.empty_cache()

        return grads

    def reshape_activations(self, layer_acts):
        return np.asarray(layer_acts).squeeze()

    @abstractmethod
    def label_to_id(self, label):
        pass

    def run_examples(self, examples, bottleneck_name):

        global bn_activation
        bn_activation = None

        def save_activation_hook(mod, inp, out):
            global bn_activation
            bn_activation = out

        handle = self.model._modules[bottleneck_name].register_forward_hook(save_activation_hook)

        self.model.to(self.model.device)
        self.model.eval()
        
        
        try:
            inputs = examples.to(self.model.device)
            self.model(inputs)
            acts = bn_activation.detach().cpu().numpy()
            del inputs
            handle.remove()
            torch.cuda.empty_cache()
        
        except Exception as e:
            print("Error:", e)
            print("Example shape:", examples.shape)
            #print("Example:", examples)
            print("bn_activation:", bn_activation)
            exit()        

        return acts
    
    
class EEGWrapper(ModelWrapper) : 
    """Wrapper base class for eeg models."""
    def __init__(self, eeg_shape, eeg_labels):
        super(ModelWrapper, self).__init__()
        # shape of the input eeg in this model
        self.eeg_shape = eeg_shape
        self.labels = eeg_labels

    def get_eeg_shape(self):
        """returns the shape of an input image."""
        return self.eeg_shape

    def label_to_id(self, label):
        return self.labels.index(label)        

class BENDR_cutted(torch.nn.Module) : 
    def __init__(self, bendr, bottleneck):
        super(BENDR_cutted, self).__init__()
        names = list(bendr._modules.keys())
        layers = list(bendr.children())

        ##BENDR has the calssifer as first saved layer: 
        #switch the order of names
        lastName = names[0]
        names.append(lastName)
        names.pop(0)
        #switch the order of layers
        lastNameL = layers[0]
        layers.append(lastNameL)
        layers.pop(0)


        self.layers = torch.nn.ModuleList()
        self.layers_names = []

        bottleneck_met = False
        for name, layer in zip(names, layers):
            if name == bottleneck:
                bottleneck_met = True
                continue  # because we already have the output of the bottleneck layer
            if not bottleneck_met:
                continue

            self.layers.append(layer)
            self.layers_names.append(name)

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y

# class BENDRWrapper(EEGWrapper) : 
#     def __init__(self, labels, modelPath, sample_length_target):
#         eeg_shape = [1, 20, sample_length_target]
#         super(BENDRWrapper, self).__init__(eeg_shape=eeg_shape, eeg_labels=labels)
#         myModel = LinearHeadBENDR(targets = 2, samples = sample_length_target, channels = 20)
        
#         myModel.load(modelPath, include_classifier= True, freeze_features = False) 

#         #myModel.eval()

#         self.model = myModel

#         self.model_name = 'LinearHead_BENDR'

#     def forward(self, x):
#         return self.model.forward() # self.model.features_forward(x)

#     def get_cutted_model(self, bottleneck):
#         return BENDR_cutted(self.model, bottleneck)