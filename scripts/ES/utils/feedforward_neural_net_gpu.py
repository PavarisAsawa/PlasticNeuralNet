import numpy as np
import torch


class FeedForwardNet:
    def __init__(self, popsize, sizes):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.weights = [torch.Tensor(popsize, sizes[i], sizes[i + 1]).uniform_(-0.1, 0.1).cuda()
                            for i in range(len(sizes) - 1)]
        self.b = [torch.Tensor(popsize, sizes[i+1]).uniform_(-0.1, 0.1).cuda()
                            for i in range(len(sizes) - 1)]
        self.architecture = sizes 
        # print('Weight: ', self.weights)   


    def forward(self, pre):
        # print('pre: ', pre)

        with torch.no_grad():
            # pre = torch.from_numpy(pre)
            """
            pre: (n_in, )
            """
            for i, W in enumerate(self.weights):
                # print('post : ', torch.einsum('ij, ijk -> ik', pre, W.float()).shape)
                # print('bias : ', self.b[i].shape)
                post =  torch.tanh(torch.einsum('ij, ijk -> ik', pre, W.float())  + self.b[i])
                pre = post

        return post.detach()

    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_models_params(self):
        p = torch.cat([ params.flatten() for params in self.weights]
                     +[ params.flatten() for params in self.b])

        return p.cpu().flatten().numpy()
    
    def get_a_model_params(self):
        p = torch.cat([ params[0].flatten() for params in self.weights]
                     +[ params[0].flatten() for params in self.b])

        return p.cpu().flatten().numpy()

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        m = 0
        for i, w in enumerate(self.weights):
            pop, a, b = w.shape
            self.weights[i] = flat_params[:, m:m + a * b].reshape(pop, a, b).cuda()
            m += a * b
        
        for i, w in enumerate(self.b):
            pop, a = w.shape
            self.b[i] = flat_params[:, m:m + a].reshape(pop, a).float().cuda()
            m += a

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        m = 0
        for i, w in enumerate(self.weights):
            pop, a, b = w.shape
            self.weights[i] = flat_params[m:m + a * b].repeat(pop, 1, 1).reshape(pop, a, b).cuda()
            m += a * b 
        for i, w in enumerate(self.b):
            pop, a = w.shape
            self.b[i] = flat_params[m:m + a].repeat(pop, 1).reshape(pop, a).float().cuda()
            m += a

    def get_weights(self):
        return [w for w in self.weights]