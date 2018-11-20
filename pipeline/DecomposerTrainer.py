import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

class DecomposerTrainer:
    def __init__(self, model, loader, lr, lights_mult):
        #model to be trained
        self.model = model
        #loader that loads training data
        self.loader = loader
        #Type of loss to use, running on cuda
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #self.lights_mult = lights_mult
    def __epoch(self):
        #Tell the model we are going to train it, thus behavior inside nn.model will change
        self.model.train()
        #Have no idea about this one
        losses = pipeline.AverageMeter(3)
        #tqdm will display a prograss bar in the terminal, should be pretty cool
        progress = tqdm(total=len(self.loader.dataset))

        for ind, tensors in enumerate(self.loader):
            tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
            #Those variable stores the value we wish the model to produce given the set of data
            inp, mask, refl_targ, shape_targ, lights_targ = tensors
            #Clean the gradient calculated by loss.backward() from last iteration
            self.optimizer.zero_grad()
            #Run the neural network 
            refl_pred, depth_pred, shape_pred, lights_pred = self.model.forward(inp, mask)
            #Find error for each corresponding output
            refl_loss = self.criterion(refl_pred, refl_targ)
            depth_loss = self.criterion(depth_pred, depth_targ)
            shape_loss = self.criterion(shape_pred, shape_targ)
            lights_loss = self.criterion(lights_pred, lights_targ)

            loss = refl_loss + depth_loss + shape_loss + lights_loss # * self.lights_mult
            #Calculate gradient 
            loss.backward()
            #optimize the neural network weight
            self.optimizer.step()

            losses.update( [l.data[0] for l in [refl_loss, shape_loss, lights_loss] ])
            #Update the progress bar
            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f | %.5f | %.3f' % (refl_loss.data[0], depth_loss.data[0], shape_loss.data[0], lights_loss.data[0]) )
        print '<Train> Losses: ', losses.avgs
        return losses.avgs


    def train(self):
        # t = trange(iters)
        # for i in t:
        err = self.__epoch()
        # print 
            # t.set_description( str(err) )
    return err

if __name__ == '__main__':
    import sys
    sys.path.append('../')