import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as Datasample

import utilis as TF_tools

class MyDataset(Datasample.Dataset):
    def __init__(self, data_X, data_Y, incontext_len, input_dim):
        self.data_X = data_X
        self.data_Y = data_Y
        self.incontext_len = incontext_len
        self.input_dim = input_dim

        self.batch_X = torch.zeros(size=(self.incontext_len+1, 2*self.input_dim))
        self.batch_Y = torch.zeros(self.input_dim)

    def __getitem__(self, index):
        
        self.batch_X = 0.0 * self.batch_X
        self.batch_Y = 0.0 * self.batch_Y
        task_index = int(index // (self.data_X.shape[1] - self.incontext_len))
        data_index = index % (self.data_X.shape[1] - self.incontext_len)

        # context data
        self.batch_X[:-1, :self.input_dim] = torch.from_numpy(self.data_X[task_index, :self.incontext_len, :]) # x
        self.batch_X[:-1, self.input_dim:] = torch.from_numpy(self.data_Y[task_index, :self.incontext_len, :]) # y
 
        # prediction data
        self.batch_X[-1, :self.input_dim] = torch.from_numpy(self.data_X[task_index, self.incontext_len+data_index, :])
        self.batch_Y[:] = torch.from_numpy(self.data_Y[task_index, self.incontext_len+data_index, :])

        return self.batch_X, self.batch_Y

    def __len__(self):
        return self.data_X.shape[0] * (self.data_X.shape[1] - self.incontext_len)
      
class TF_linear_att(nn.Module):
    def __init__(self, n, d, device="cuda:1",initial_input=False, intial_value=[]):
        #########
        #Default setting:
        # size of input_x : (batchsize, (n+1), 2d)
        #    N: tasks numbers
        #    d: dimension of matrix
        #    n: contextual length
        ##########
        # current structure only has a single attention layer
        
        super(TF_linear_att, self).__init__()
        self.n = torch.tensor(n, requires_grad=False,device=device)
        self.d = torch.tensor(d, requires_grad=False,device=device)
        self.M = torch.zeros(size=(n+1,n+1),requires_grad=False,device=device) # No gradient: (n+1) * (n+1)
        for i in range(self.n):
            self.M[i,i] = 1.0

        self.zero_d_d = torch.zeros(size=(d,d),requires_grad=False,device=device) 
        self.zero_d_2d = torch.zeros(size=(d,2*d),requires_grad=False,device=device) 
        if initial_input:
            initial_data = np.array(intial_value).astype(np.float32)
            print("input matrix with shape: ")
            print(initial_data.shape)
            initial_data_inv = np.linalg.inv(initial_data)
            self.params = nn.ParameterList([nn.Parameter(torch.tensor(initial_data)), nn.Parameter(torch.tensor(initial_data_inv))])
        else:
            self.params = nn.ParameterList([nn.Parameter(torch.eye(d)), nn.Parameter(torch.eye(d))])

    def attention(self, input_x, mask=True):
        # P, Q : R^{2d * 2d}
        # size of input_x : (batchsize, (n+1), 2d)
        
        P_full =  torch.cat([self.zero_d_2d, torch.cat([self.zero_d_d, self.params[0]], dim=1)], dim=0)
        Q_full =  torch.cat([torch.cat([self.params[1], self.zero_d_d], dim=0), self.zero_d_2d.T], dim=1)
        output = torch.einsum("ab,bcd->acd", Q_full, input_x.permute((2,1,0))) # (2d, n+1, b)
        output = torch.einsum("abc,bdc->adc", input_x.permute((1,2,0)), output)  # (n+1, n+1, b)
        if mask:
            output = torch.einsum("ab,bcd->acd", self.M, output) # (n+1, n+1, b)
        output = torch.einsum("abc,bdc->adc", input_x.permute((2,1,0)), output)  # (2d, n+1, b)
        output = torch.einsum("ab,bcd->acd", P_full, output) # (2d, n+1, b)
        output = input_x + output.permute((2,1,0)) / self.n
        return output

    def forward(self, input_x, mask=True):
        output = self.attention(input_x, mask=mask)
        output = output[:, -1, self.d:]
        return output

def my_loss(y_pred, y):
    # y shape: (batch, d)
    err = y_pred - y
    err = torch.sum(err ** 2,axis=1)
    err = torch.mean(err)
    return err

def train_system(n, d, batch_size, train_valid_X_Y, best_model_path, device="cuda:2", epoch_all=20,lr=1,input_initial=False, initial_matrix=[]):
    train_X, train_Y, valid_X, valid_Y = train_valid_X_Y
    train_loader = DataLoader(MyDataset(train_X, train_Y, n, d), batch_size=batch_size, shuffle=True)#, num_workers=2)
    valid_loader = DataLoader(MyDataset(valid_X, valid_Y, n, d), batch_size=batch_size, shuffle=False)#, num_workers=2)
    
   
    if input_initial:
        model = TF_linear_att(n, d, device=device, initial_input=True, intial_value=initial_matrix).to(device)
    else:
        model = TF_linear_att(n, d, device=device,).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_mse_hist = []
    valid_mse_hist = []
    param_list_P = []
    param_list_Q = []
    valid_loss_max = 100000000000
    patience_total = 200
    for epoch in range(epoch_all):
        global_step = 0
        model.train()
        train_loss_array = []
        for i, data_temp in enumerate(train_loader):
            train_X_temp, train_Y_temp = data_temp
            train_X_temp = train_X_temp.to(device)
            train_Y_temp = train_Y_temp.to(device)
            optimizer.zero_grad()
            pred_Y = model(train_X_temp)
            
            loss = my_loss(pred_Y, train_Y_temp)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step +=1
            if global_step % 100 == 0:
                str_output = "Epoch: "+ str(epoch) + "|| [" + str(i) + "/" + str(len(train_loader) ) +"]" + " train loss: " + str(np.mean(train_loss_array))
                print(str_output)

        model.eval()
        valid_loss_array = []
        for i, data_temp in enumerate(valid_loader):

            valid_X_temp, valid_Y_temp = data_temp
            valid_X_temp = valid_X_temp.to(device)
            valid_Y_temp = valid_Y_temp.to(device)

            pred_Y = model(valid_X_temp)
            loss = my_loss(pred_Y, valid_Y_temp)
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        train_mse_hist.append(train_loss_cur)
        valid_mse_hist.append(valid_loss_cur)
        str_output = "Epoch: " + str(epoch) + " || ValMSE: " + str(valid_loss_cur) + ", TraMSE: "+ str(train_loss_cur)
        print(str_output)

        if valid_loss_cur < valid_loss_max:
            valid_loss_max = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), best_model_path)
            print("best model saved! index: " + str(epoch))
        else:
            bad_epoch += 1
            if bad_epoch >= patience_total:
                print("Epoch {} ends ...".format(epoch))
                break
    return train_mse_hist, valid_mse_hist, model #, param_list_P, param_list_Q

def predict(n, d, device, test_X, test_Y, best_model_path, batchsize=1):
    test_loader = DataLoader(MyDataset(test_X, test_Y, n, d), batch_size=batchsize, shuffle=False)#, num_workers=2)

    model = TF_linear_att(n, d, device=device).to(device)
    print("Start loading")
    model.load_state_dict(torch.load(best_model_path))
    
    model.eval()
    pred_Y_list = []
    target_Y_list = []
    test_loss_array = [] 
    
    for i, data_temp in enumerate(test_loader):

        test_X_temp, test_Y_temp = data_temp
        test_X_temp = test_X_temp.to(device)
        test_Y_temp = test_Y_temp.to(device)

        pred_Y = model(test_X_temp)
        loss = my_loss(pred_Y, test_Y_temp)
        test_loss_array.append(loss.item())
        pred_Y_list.append(pred_Y.detach().cpu())
        target_Y_list.append(test_Y_temp.detach().cpu())

    test_loss_cur = np.mean(test_loss_array)
    return np.concatenate(pred_Y_list, axis=0), np.concatenate(target_Y_list,axis=0), test_loss_cur, model