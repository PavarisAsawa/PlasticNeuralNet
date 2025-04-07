import torch

class SeqLSTMs():

    def __init__(self, popsize, arch): #in_channels, hid_size, out_channels):
        super(SeqLSTMs, self).__init__()


        self.arch = arch
        in_channels = arch[0]
        hid_size = arch[1]
        out_channels = arch[2]
        
        self.popsize = popsize


        arch_base = tuple([in_channels, hid_size, in_channels])
        arch_final = tuple([in_channels, hid_size, out_channels])

        self.model_1 = LSTMs(popsize, arch_base)
        self.model_2 = LSTMs(popsize, arch_base)
        self.model_3 = LSTMs(popsize, arch_final)

        self.n_params_b = self.model_1.get_n_params()
        self.n_params_f = self.model_3.get_n_params()

        init_w = 0.1
        init_params_b = torch.Tensor(popsize, self.n_params_b).uniform_(-init_w, init_w)
        init_params_f = torch.Tensor(popsize, self.n_params_f).uniform_(-init_w, init_w)
        print('init_w: ', init_w)

        self.model_1.set_params(init_params_b)
        self.model_2.set_params(init_params_b)
        self.model_3.set_params(init_params_f)
        

    def forward(self, inp):
        out_1 = self.model_1.forward(inp)
        out_2 = self.model_2.forward(out_1)
        out_3 = self.model_3.forward(out_2)

        return out_3.squeeze_()
    
    def set_params(self, pop):
        m = 0
        popsize = pop.shape[0]

        self.model_1.set_params(torch.Tensor(pop[:, m:m+self.n_params_b]))
        m += self.n_params_b
        self.model_2.set_params(torch.Tensor(pop[:, m:m+self.n_params_b]))
        m += self.n_params_b
        self.model_3.set_params(torch.Tensor(pop[:, m:m+self.n_params_f]))
        m += self.n_params_f


    def get_n_params(self):
        
        return self.n_params_b + self.n_params_b + self.n_params_f

    def get_params_a_model(self):
        p = torch.cat([torch.Tensor(self.model_1.get_params_a_model())]  
                     +[torch.Tensor(self.model_2.get_params_a_model())] 
                     +[torch.Tensor(self.model_3.get_params_a_model())]
                     )
        return p.flatten()


class LSTMs():

    def __init__(self, popsize, arch): #in_channels, hid_size, out_channels):
        super(LSTMs, self).__init__()


        self.arch = arch
        in_channels = arch[0]
        hid_size = arch[1]
        out_channels = arch[2]
        
        self.popsize = popsize

        self.in_channels = in_channels
        self.hid_size = hid_size
        self.out_channels = out_channels

        init_hidd = 0.1
        self.hidden_state = torch.Tensor(popsize, hid_size, 1).uniform_(-init_hidd, init_hidd).cuda()
        self.cell_state = torch.Tensor(popsize, hid_size, 1).uniform_(-init_hidd, init_hidd).cuda()

        self.n_params = self.get_n_params_a_model()
        init_params_b = torch.Tensor(popsize, self.n_params).uniform_(-init_hidd, init_hidd)
        print('init_hidd: ', init_hidd)
        self.set_models_params(init_params_b.numpy())

    def forward(self, inp):
        with torch.no_grad():        

            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)
            # print('inp', inp.shape)
            # print('self.hidden_state', self.hidden_state.shape)
            # print('x', x.shape)
            # print('self.Wf', self.Wf.shape)
            # print('self.Bf', self.Bf.shape)

            f = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wf.float(), x.float())+self.Bf.float())
            i = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wi.float(), x.float())+self.Bi.float())

            c = torch.tanh( torch.einsum('lbn,lbc->lnc', self.Wc.float(), x.float())+self.Bc.float())

            self.cell_state = f * self.cell_state + i * c

            o = torch.sigmoid( torch.einsum('lbn,lbc->lnc', self.Wo.float(), x.float())+self.Bo.float())

            self.hidden_state = o * torch.tanh(self.cell_state)

            x = torch.cat((inp.unsqueeze(-1), self.hidden_state), dim=1)
            out = torch.tanh( torch.einsum('lbn,lbc->lnc', self.Wout.float(), x.float())) ## (popsize, out_size, 1)

        return out.squeeze_()

    def get_n_params_a_model(self):
        n_i, n_h, n_o = self.arch
        return (n_i+n_h)*n_h * 4 + (n_i+n_h)*n_o + n_h * 4

    def get_models_params(self):
        p = torch.cat([ self.Wf.flatten()]  
                     +[ self.Wi.flatten()] 
                     +[ self.Wc.flatten()]
                     +[ self.Wo.flatten()]
                     +[ self.Wout.flatten()]
                     +[ self.Bf.flatten()]
                     +[ self.Bi.flatten()]
                     +[ self.Bc.flatten()]
                     +[ self.Bo.flatten()]
                     )
        return p.flatten().cpu().numpy()

    def get_a_model_params(self):
        p = torch.cat([ self.Wf[0].flatten()]  
                     +[ self.Wi[0].flatten()] 
                     +[ self.Wc[0].flatten()]
                     +[ self.Wo[0].flatten()]
                     +[ self.Wout[0].flatten()]
                     +[ self.Bf[0].flatten()]
                     +[ self.Bi[0].flatten()]
                     +[ self.Bc[0].flatten()]
                     +[ self.Bo[0].flatten()]
                     )
        return p.flatten().cpu().numpy()
    

    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
        
        ##population shape: (popsize, n_total_params)

        n_i, n_h, n_o = self.arch

        m = 0
        self.Wf = flat_params[:,m:m+(n_i+n_h)*n_h].reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wi = flat_params[:,m:m+(n_i+n_h)*n_h].reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wc = flat_params[:,m:m+(n_i+n_h)*n_h].reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wo = flat_params[:,m:m+(n_i+n_h)*n_h].reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wout = flat_params[:,m:m+(n_i+n_h)*n_o].reshape(self.popsize, n_i+n_h, n_o).cuda()
        m += (n_i+n_h)*n_o

        # print('self.Wf: ', self.Wf.shape)
        # print('self.Wi: ', self.Wi.shape)
        # print('self.Wc: ', self.Wc.shape)
        # print('self.Wo: ', self.Wo.shape)
        # print('self.Wout: ', self.Wout.shape)

        self.Bf = flat_params[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bi = flat_params[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bc = flat_params[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h
        self.Bo = flat_params[:,m:m+n_h].unsqueeze(-1).cuda()
        m += n_h

        # print('self.Bf: ', self.Bf.shape)
        # print('self.Bi: ', self.Bi.shape)
        # print('self.Bc: ', self.Bc.shape)
        # print('self.Bo: ', self.Bo.shape)

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params)
       
        ##population shape: (popsize, n_total_params)

        n_i, n_h, n_o = self.arch

        m = 0
        self.Wf = flat_params[m:m+(n_i+n_h)*n_h].repeat(self.popsize, 1, 1).reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wi = flat_params[m:m+(n_i+n_h)*n_h].repeat(self.popsize, 1, 1).reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wc = flat_params[m:m+(n_i+n_h)*n_h].repeat(self.popsize, 1, 1).reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wo = flat_params[m:m+(n_i+n_h)*n_h].repeat(self.popsize, 1, 1).reshape(self.popsize, n_i+n_h, n_h).cuda()
        m += (n_i+n_h)*n_h
        self.Wout = flat_params[m:m+(n_i+n_h)*n_o].repeat(self.popsize, 1, 1).reshape(self.popsize, n_i+n_h, n_o).cuda()
        m += (n_i+n_h)*n_o

        self.Bf = flat_params[m:m+n_h].repeat(self.popsize, 1).unsqueeze(-1).cuda()
        m += n_h
        self.Bi = flat_params[m:m+n_h].repeat(self.popsize, 1).unsqueeze(-1).cuda()
        m += n_h
        self.Bc = flat_params[m:m+n_h].repeat(self.popsize, 1).unsqueeze(-1).cuda()
        m += n_h
        self.Bo = flat_params[m:m+n_h].repeat(self.popsize, 1).unsqueeze(-1).cuda()
        m += n_h

        # print('self.Wf: ', self.Wf.shape)
        # print('self.Wi: ', self.Wi.shape)
        # print('self.Wc: ', self.Wc.shape)
        # print('self.Wo: ', self.Wo.shape)
        # print('self.Wout: ', self.Wout.shape)
        # print('self.Bf: ', self.Bf.shape)
        # print('self.Bi: ', self.Bi.shape)
        # print('self.Bc: ', self.Bc.shape)
        # print('self.Bo: ', self.Bo.shape)