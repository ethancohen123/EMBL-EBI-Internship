# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:44:58 2020

@author: ethan
"""


import torch
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import copy
import time
from metrics import calc_loss
from metrics import compute_metrics,print_metrics,metrics_line,normalise_mask_set


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Trainers(object):

    def __init__(self, model, optimizer=None, scheduler=None):

        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.optimizer = optimizer
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = scheduler
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.1)

    def train_model(self, dataloaders, num_epochs=25):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        epochs_metrics = {
            'train': [],
            'val': []
        }

        for epoch in range(num_epochs):
            print('Epoch {}/{}:'.format(epoch+1, num_epochs))

            since = time.time()


            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    for param_group in self.optimizer.param_groups:
                        print("\tlearning rate: {:.2e}".format(
                            param_group['lr']))

                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in dataloaders[phase]:
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs=inputs.permute(0,1,3,2)#put the mask and inputs in the same settings
                    

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = self.model(inputs)
                        
                        loss = calc_loss(outputs, labels,metrics)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            

                    # statistics
                    epoch_samples += inputs.size(0)

                computed_metrics = compute_metrics(metrics, epoch_samples)
                print_metrics(computed_metrics, phase)
                epochs_metrics[phase].append(computed_metrics)
                epoch_loss = metrics['loss'] / epoch_samples

                if phase == 'train':
                    self.scheduler.step()

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print("\tSaving best model, epoch loss {:4f} < best loss {:4f}".format(
                        epoch_loss, best_loss))
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            time_elapsed = time.time() - since
            print('\t{:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('-' * 10)

        print('Best val loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        metrics_line(epochs_metrics)
   
    
    
    def predict(self, X):
        
        self.model.eval()
        X=X.permute(0,1,3,2)
        inputs = X.to(self.device)
        pred = self.model(inputs)
        
        avant_norm = pred.data.cpu().numpy()
        
       
        

        return avant_norm
    