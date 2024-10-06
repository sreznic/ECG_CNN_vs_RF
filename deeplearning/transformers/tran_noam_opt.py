from tran_model import CTN

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
if __name__ == "__main__":
    from tran_train import d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz
    import torch.optim

    model = CTN(d_model, nhead, d_ff, num_layers, dropout_rate, deepfeat_sz, [i for i in range(3)])
    opt = NoamOpt(d_model, 1, 1, torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9))
    rates = []
    for i in (range(1000)):
        opt.step()
        rates.append(opt.rate())
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.arange(1000), rates)
    plt.show()
    