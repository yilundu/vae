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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default',
    'name of experiment')
flags.DEFINE_string('dataset', 'frey',
    'what dataset to use (mnist or frey)')
flags.DEFINE_integer('resume_iter', 0,
    'resume value')
flags.DEFINE_integer('num_iter', int(1e6),
    'number of iterations of training')
flags.DEFINE_integer('batch_size', 100,
    'batch size to use during training')
flags.DEFINE_bool('train', True,
    'number of epochs')
flags.DEFINE_integer('latent_dim', 5,
    'dimensionality of latent dim')
flags.DEFINE_integer('hidden_dim', 400,
    'number of hidden dimensions')
flags.DEFINE_bool('gen_plots', True,
    'generate loss curves over iterations')

class FreyFaces(Dataset):

    def __init__(self, train=True):
        dat = loadmat("data/frey_rawface.mat")["ff"]
        dat = dat.transpose((1, 0)).reshape((-1, 28, 20))
        self.dat = dat / 255.
        dat = self.dat

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

    def __init__(self, hidden_dim=20, input_dim=784, nh=400, prob_dist="discrete"):
        super(VAE, self).__init__()
        self.encode_fc = nn.Linear(input_dim, nh)
        self.encode_output = nn.Linear(nh, 2 * hidden_dim)
        self.prob_dist = prob_dist

        self.decode_fc = nn.Linear(hidden_dim, nh)

        if self.prob_dist == "discrete":
            self.decode_output = nn.Linear(nh, input_dim)
        elif self.prob_dist == "continuous":
            self.decode_output = nn.Linear(nh, 2 * input_dim)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

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

        if self.prob_dist == "discrete":
            output = F.sigmoid(output)
        elif self.prob_dist == "continuous":
            output_mean, output_log_var = output[:, :self.input_dim], output[:, self.input_dim:]
            output_mean = F.sigmoid(output_mean)
            output = torch.cat([output_mean, output_log_var], dim=1)

        return (mean, log_var), output

    def compute_loss(self, output, inp, loss_type):
        (mean, log_var), output = output
        kl_loss = (-0.5 * (1 + log_var - mean.pow(2) - log_var.exp())).sum(dim=1).mean(dim=0)

        if loss_type == "bce":
            prob_loss = (F.binary_cross_entropy(output, inp, reduction='none').sum(dim=1).mean(dim=0))
        elif loss_type == "gauss":
            mean, log_var = output[:, :self.input_dim], output[:, self.input_dim:]
            exp_term = ((inp - mean).pow(2) / (2 * log_var.exp())).sum(dim=1).mean(dim=0)
            norm_term = 0.5 * (log_var.sum(dim=1)).mean(dim=0)
            prob_loss = exp_term + norm_term

        return kl_loss + prob_loss

    def generate_sample(self):
        z = torch.randn(64, self.hidden_dim).cuda()
        decode_hidden = F.relu(self.decode_fc(z))
        output = self.decode_output(decode_hidden)

        if self.prob_dist == "discrete":
            output = F.sigmoid(output)
        elif self.prob_dist == "continuous":
            mean, log_var = output[:, :self.input_dim], output[:, self.input_dim:]
            # output = mean + torch.randn_like(log_var).cuda() * torch.exp(0.5 * log_var)
            output = mean

        ouput = torch.clamp(output, 0.0, 1.0)

        return output


    def sample_posterior(self, x, batch=256):
        """Approximates sampling from p(z|x) using Langevin dynamics"""
        for i in range(100):
            pass


    def estimate_prob(self, inp):
        """Returns to a Monte Carlo estimate of the negative log likelihood of samples"""



def main():

    if FLAGS.dataset == "mnist":
        train_dataloader = DataLoader(MNIST("/root/data", train=True, download=True, transform=transforms.ToTensor()), batch_size=FLAGS.batch_size)
        test_dataloader = DataLoader(MNIST("/root/data", train=False, download=True, transform=transforms.ToTensor()), batch_size=FLAGS.batch_size)
        input_dim = 784
        prob_dist = "discrete"
    else:
        train_dataloader = DataLoader(FreyFaces(train=True), batch_size=32)
        test_dataloader = DataLoader(FreyFaces(train=False), batch_size=32)
        input_dim = 560
        prob_dist = "continuous"


    model = VAE(hidden_dim=FLAGS.latent_dim, input_dim=input_dim, nh=FLAGS.hidden_dim, prob_dist=prob_dist).train().cuda()
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    logger = TensorBoardOutputFormat(logdir)

    it = FLAGS.resume_iter

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_{}".format(FLAGS.resume_iter))
        model.load_state_dict(torch.load(model_path))


    if FLAGS.train:

        stop = False
        its = []
        train_losses = []
        test_losses = []
        test_dataloader_iter = iter(test_dataloader)

        while it < FLAGS.num_iter:
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

                if it % (10 * FLAGS.batch_size) == 0:
                    loss = loss.item()
                    logger.writekvs({"loss": loss})
                    print(it, loss)
                    if FLAGS.gen_plots:
                        its.append(it)
                        train_losses.append(-1 * loss)

                        try:
                            dat, label = test_dataloader_iter.next()
                        except:
                            test_dataloader_iter = iter(test_dataloader)
                            dat, label = test_dataloader_iter.next()

                        if FLAGS.dataset == "mnist":
                            dat = dat.cuda().reshape((dat.size(0), 28*28))
                        else:
                            dat = dat.float().cuda().reshape((dat.size(0), 28*20))

                        outputs = model.forward(dat)

                        if FLAGS.dataset == "mnist":
                            loss = model.compute_loss(outputs, dat, "bce")
                        else:
                            loss = model.compute_loss(outputs, dat, "gauss")

                        test_losses.append(-1 * loss)


                it += FLAGS.batch_size

                if it > FLAGS.num_iter:
                    break

        if FLAGS.gen_plots:
            plt.semilogx(its, train_losses, "r")
            plt.semilogx(its, test_losses, "b")
            plt.ylabel("ELBO")

            if FLAGS.dataset == "frey":
                data_string = "Frey Faces"
            elif FLAGS.dataset == "mnist":
                data_string = "MNIST"
            plt.title("{}, $N_z = {}$".format(data_string, FLAGS.latent_dim))
            time = str(datetime.datetime.now())
            plt.savfig("plot_{}_{}_{}.png".format(FLAGS.dataset, time, FLAGS.latent_dim))

        model_path = osp.join(logdir, "model_{}".format(it))
        torch.save(model.state_dict(), model_path)

    output = model.generate_sample()
    output = output.cpu().detach().numpy()

    if FLAGS.dataset == "mnist":
        output = output.reshape((8, 8, 28, 28)).transpose((0, 2, 1, 3)).reshape((8*28, 8*28))
    elif FLAGS.dataset == "frey":
        output = output.reshape((8, 8, 28, 20)).transpose((0, 2, 1, 3)).reshape((8*28, 8*20))

    time = str(datetime.datetime.now())
    imsave("test_{}_{}_{}.png".format(FLAGS.dataset, time, FLAGS.latent_dim), output)

    print("Done")

if __name__ == "__main__":
    main()
