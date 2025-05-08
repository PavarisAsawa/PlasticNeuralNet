import numpy as np
import torch

def var_norm(w, eps=1e-5):
    # method_0: devide by variance
    mean = torch.mean(input=w, dim=[1,2], keepdim=True)
    var = torch.var(input=w, dim=[1,2], keepdim=True)
    w = (w - mean) / torch.sqrt(var + eps)
    return w

def max_norm(w, eps=1e-5):
    # method_1: devide by max value
    max_val = torch.max(torch.abs(w).flatten(start_dim=1, end_dim=2), dim=1)
    max_val = max_val[0].unsqueeze(1).unsqueeze(2)    # print('w - mean: ', w - mean)
    w = w / max_val
    return w

def clip_weights(w, eps=1e-5):
    # method_2: clip weight higher than 1.0
    w = torch.clamp(w, min=-1.0, max=1.0)
    return w

class HebbianNet:
    def __init__(self, 
                 popsize, 
                 sizes, 
                 init_noise=0.04,  # 0.02
                 norm_mode='var'):
        """
        sizes: [input_size, hid_1, ..., output_size]

        Alternative way: initialize weight as uniform distribution
        self.A = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(-0.01, 0.01)
                            for i in range(len(sizes) - 1)]

        """
        # initial weight uniform dist range (-0.1, 0.1)
        self.architecture = sizes
        self.weights = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(-init_noise, init_noise).cuda()
                            for i in range(len(sizes) - 1)]
        # self.b = [torch.Tensor(popsize, sizes[i+1]).uniform_(-0.01, 0.01).cuda()
        #                     for i in range(len(sizes) - 1)]
        self.one_array = [torch.ones(popsize, sizes[i], sizes[i + 1]).cuda()
                            for i in range(len(sizes) - 1)]
        # print('self.one_array', self.one_array)

        self.A = self.initialize_weights(popsize, sizes)
        self.B = self.initialize_weights(popsize, sizes)
        self.C = self.initialize_weights(popsize, sizes)
        self.D = self.initialize_weights(popsize, sizes)
        self.lr = self.initialize_weights(popsize, sizes)

        if norm_mode == 'var':
            self.WeightStand = var_norm
        elif norm_mode == 'max':
            self.WeightStand = max_norm
        elif norm_mode == 'clip':
            self.WeightStand = clip_weights

    def forward(self, pre):

        with torch.no_grad():
            """
            pre: (n_in, )
            """
            for i, W in enumerate(self.weights):
                post =  torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float()))
                # post =  torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float()) + self.b[i])
                self.weights[i] = self.hebbian_update(i, W, pre, post, self.A[i], self.B[i], self.C[i], self.D[i], self.lr[i])
                pre = post

        return post.float().detach()

    def initialize_weights(self, popsize, sizes):
        return [torch.normal(0, 0.025, (popsize, sizes[i], sizes[i + 1])).cuda() for i in range(len(sizes) - 1)]

    def hebbian_update(self, hid_num ,weights, pre, post, A, B, C, D, lr):

        i = self.one_array[hid_num] * pre.unsqueeze(2)
        j = post.unsqueeze(2).expand(-1,-1, weights.shape[1]).transpose(1,2)
        ij = i * j

        weights = weights + lr * (A*ij + B*i + C*j + D)
        weights = self.WeightStand(weights)

        return weights

    def get_weights(self):
        return [w for w in self.weights]
    
    def get_n_params_a_model(self):
        return len(self.get_a_model_params())
    
    def get_models_params(self):
        all_params = [self.A, self.B, self.C, self.D, self.lr]
        p = torch.cat([param.flatten() for params in all_params for param in params])
        return p.flatten().cpu().numpy()

    def get_a_model_params(self):
        all_params = [self.A, self.B, self.C, self.D, self.lr]
        p = torch.cat([param[0].flatten() for params in all_params for param in params])
        return p.flatten().cpu().numpy()

    def update_params(self, hebb_list, flat_params, start_index):
        m = start_index
        for i, hebb in enumerate(hebb_list):
            pop, a, b = hebb.shape
            hebb_list[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b
        return m

    def update_a_model_params(self, hebb_list, flat_params, start_index):
        m = start_index
        for i, hebb in enumerate(hebb_list):
            pop, a, b = hebb.shape
            hebb_list[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 
        return m

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params.shape)

        m = 0
        m = self.update_params(self.A, flat_params, m)
        m = self.update_params(self.B, flat_params, m)
        m = self.update_params(self.C, flat_params, m)
        m = self.update_params(self.D, flat_params, m)
        m = self.update_params(self.lr, flat_params, m)
        # for i, w in enumerate(self.b):
        #     pop, a = w.shape
        #     self.b[i] = flat_params[:, m:m + a].reshape(pop, a).float().cuda()
        #     m += a

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        # print('flat_params: ', flat_params)

        m = 0
        m = self.update_a_model_params(self.A, flat_params, m)
        m = self.update_a_model_params(self.B, flat_params, m)
        m = self.update_a_model_params(self.C, flat_params, m)
        m = self.update_a_model_params(self.D, flat_params, m)
        m = self.update_a_model_params(self.lr, flat_params, m)
        # for i, w in enumerate(self.b):
        #     pop, a = w.shape
        #     self.b[i] = flat_params[m:m + a].repeat(pop, 1).reshape(pop, a).float().cuda()
        #     m += a