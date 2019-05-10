from torch.utils.data import DataLoader
import torch
import os.path as osp
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch import optim
from scipy.misc import imsave
from tensorflow.python.platform import flags
from baselines.logger import TensorBoardOutputFormat
from torch.utils.data import Dataset
from scipy.io import loadmat
import datetime

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default',
    'name of experiment')
flags.DEFINE_string('dataset', 'frey',
    'what dataset to use (mnist or frey)')
flags.DEFINE_integer('resume_iter', -1,
    'resume value')
flags.DEFINE_integer('num_epochs', 10,
    'number of epochs')
flags.DEFINE_bool('train', True,
    'number of epochs')

class FreyFaces(Dataset):

    def __init__(self, train=True):
        dat = loadmat("data/frey_rawface.mat")["ff"]
        dat = dat.transpose((1, 0)).reshape((-1, 28, 20))
        self.dat = dat / 255.

        batch = dat.shape[0]
        idx = int(0.9 * batch)

        if train:
            self.dat = self.dat[:idx]
        else:
            self.dat = self.dat[idx:]

    def __len__(self):
        return self.dat.shape[0]

    def __getitem__(self, idx):
        return self.dat[idx], 0


class VAE(nn.Module):

    def __init__(self, hidden_dim=20, input_dim=784):
        super(VAE, self).__init__()
        self.encode_fc = nn.Linear(input_dim, 400)
        self.encode_output = nn.Linear(400, 2*hidden_dim)

        self.decode_fc = nn.Linear(hidden_dim, 400)
        self.decode_output = nn.Linear(400, input_dim)

        self.hidden_dim = hidden_dim

    def forward(self, inp):
        fc1 = F.relu(self.encode_fc(inp))
        encode = self.encode_output(fc1)

        log_var, mean = encode[:, :self.hidden_dim], encode[:, self.hidden_dim:]
        z = torch.randn_like(log_var).cuda() * torch.exp(0.5 * log_var) + mean

        # Don't backpropogate gradients to the earlier network'
        # z = z.detach()

        # Paper specifies Tanh
        decode_hidden = F.relu(self.decode_fc(z))
        output = self.decode_output(decode_hidden)
        output = F.sigmoid(output)

        return (mean, log_var), output

    def compute_loss(self, output, inp, loss_type):
        (mean, log_var), output = output
        kl_loss = (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum()

        if loss_type == "bce":
            bce_loss = (F.binary_cross_entropy(output, inp, reduction='none').sum(dim=1).mean(dim=0))
        elif loss_type == "gauss":
            bce_loss = (output - inp).pow(2).sum(dim=1).mean(dim=0) * 100

        return kl_loss + bce_loss

    def generate_sample(self):
        z = torch.randn(64, self.hidden_dim).cuda()
        decode_hidden = F.relu(self.decode_fc(z))
        output = self.decode_output(decode_hidden)
        output = F.sigmoid(output)

        return output



def main():

    if FLAGS.dataset == "mnist":
        train_dataloader = DataLoader(MNIST("/root/data", train=True, download=True, transform=transforms.ToTensor()), batch_size=32)
        test_dataloader = DataLoader(MNIST("/root/data", train=False, download=True, transform=transforms.ToTensor()), batch_size=32)
        input_dim = 784
    else:
        train_dataloader = DataLoader(FreyFaces(train=True), batch_size=32)
        test_dataloader = DataLoader(FreyFaces(train=False), batch_size=32)
        input_dim = 560


    model = VAE(hidden_dim=5, input_dim=input_dim).train().cuda()
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    logger = TensorBoardOutputFormat(logdir)

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.resume_iter != -1:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        model.load_state_dict(torch.load(model_path))


    if FLAGS.train:
        for i in range(FLAGS.num_epochs):
            for dat, label in tqdm(train_dataloader):

                if FLAGS.dataset == "mnist":
                    dat = dat.cuda().reshape((dat.size(0), 28*28))
                else:
                    dat = dat.float().cuda().reshape((dat.size(0), 28*20))

                optimizer.zero_grad()
                outputs = model.forward(dat)

                if FLAGS.dataset == "mnist":
                    loss = model.compute_loss(outputs, dat, "bce")
                else:
                    loss = model.compute_loss(outputs, dat, "gauss")

                loss.backward()
                optimizer.step()

                loss = loss.item()
                logger.writekvs({"loss": loss})

                if it % 10 == 0:
                    print(loss)

                if it % 1000 == 0:
                    model_path = osp.join(logdir, "model_{}".format(it))
                    torch.save(model.state_dict(), model_path)

                it += 1

    output = model.generate_sample()
    output = output.cpu().detach().numpy()

    if FLAGS.dataset == "mnist":
        output = output.reshape((8, 8, 28, 28)).transpose((0, 2, 1, 3)).reshape((8*28, 8*28))
    elif FLAGS.dataset == "frey":
        output = output.reshape((8, 8, 28, 20)).transpose((0, 2, 1, 3)).reshape((8*28, 8*20))

    time = str(datetime.datetime.now())
    imsave("test_{}_{}.png".format(FLAGS.dataset, time), output)

    print("Done")

if __name__ == "__main__":
    main()
