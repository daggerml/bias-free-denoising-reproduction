from flax.metrics.tensorboard import SummaryWriter
import os
import math
from ml_collections import ConfigDict
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from bfdn.models import DnCNN
from bfdn.data import Data


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data \
            .normal_(mean=0, std=math.sqrt(2. / 9. / 64.)) \
            .clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)
    return


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i], Img[i],
                                        data_range=data_range)
    return (PSNR/Img.shape[0])


def main(conf: ConfigDict, out_loc, extra_images=[]):
    writer = SummaryWriter(out_loc)
    writer.hparams(dict(conf))
    torch.manual_seed(conf.seed)
    rng = np.random.default_rng(conf.seed + 1)
    if not os.path.isdir(out_loc):
        print('making directory:', out_loc)
        os.mkdir(out_loc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = Data(train=True)
    valid_set = Data(train=False)
    print('# of training samples:', len(train_set))
    train_iter = DataLoader(dataset=train_set,
                            num_workers=4,
                            batch_size=conf.batch_size,
                            shuffle=True)
    from PIL import Image
    extras = {k: np.array(Image.open(f'data/extra/{k}'), dtype=np.float32)
              for k in extra_images}
    extras = {k: v.mean(axis=-1, keepdims=True) for k, v in extras.items()}
    extras = {k: np.transpose(v, (2, 0, 1)) for k, v in extras.items()}
    net = DnCNN(channels=1, num_of_layers=conf.num_layers, bias=conf.use_bias)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()
    criterion.to(device)
    model = nn.DataParallel(net, device_ids=[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    for epoch in range(conf.num_epochs):
        if epoch == conf.milestone:
            print('epoch:', epoch, 'learning rate', conf.learning_rate / 10.0)
            for param_group in optimizer.param_groups:
                param_group["lr"] = conf.learning_rate / 10.0
        model.train()
        for i, data in enumerate(train_iter):
            model.zero_grad()
            optimizer.zero_grad()
            data.to(device)
            shape = data.shape
            noise = torch.zeros(shape)
            stds = rng.uniform(conf.noise_low, conf.noise_high, size=shape[0])
            for j, s in enumerate(stds):
                noise[j, ...] = torch.FloatTensor(shape[1:])\
                    .normal_(mean=0, std=s/255.)
            noise.to(device)
            X_train = (data + noise).to(device)
            pred_noise = model(X_train)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            # results
            recon = torch.clamp(X_train - pred_noise, 0., 1.)
            psnr_train = batch_PSNR(recon, data, 1.)
            print("[epoch %d][%4d/%4d] loss: %.4f PSNR_train: %.4f" %
                  (epoch+1, i+1, len(train_set), loss.item(), psnr_train))
            if i % 10 == 0:
                writer.scalar('train loss', loss.item(), epoch)
                writer.scalar('train psnr', psnr_train, epoch)
            if conf.debug and (i >= 19):
                break
        model.eval()
        for k in range(5):
            data = torch.unsqueeze(train_set[10_000 * k], 0)
            X = data + torch.FloatTensor(data.size())\
                .normal_(mean=0, std=(conf.noise_low + conf.noise_high)/(2 * 255.))
            recon = torch.clamp(X - model(X), 0., 1.)
            writer.image(
                f'train/{k}_clean:noisy:recon',
                make_grid([data[0], X[0], recon[0]]),
                epoch
            )
        psnr_val = 0
        loss = 0
        for k, data in enumerate(valid_set):
            data = torch.unsqueeze(data, 0)
            noise = torch.FloatTensor(data.size()).normal_(mean=0, std=conf.valid_noise/255.)
            X_valid = (data + noise).to(device)
            pred_noise = model(X_valid)
            loss += criterion(pred_noise, noise).item()
            recon = torch.clamp(data - pred_noise, 0., 1.)
            psnr_val += batch_PSNR(recon, data, 1.)
            if k < 5:
                writer.image(
                    f'valid/{k}_clean:noisy:recon',
                    make_grid([data[0], X_valid[0], recon[0]]),
                    epoch
                )

        psnr_val /= len(valid_set)
        loss /= len(valid_set)
        print("[epoch %d] valid loss: %.4f PSNR_val: %.4f" % (epoch+1, loss, psnr_val))
        writer.scalar('valid loss', loss, epoch)
        writer.scalar('valid psnr', psnr_val, epoch)
        for k, data in extras.items():
            X_valid = torch.unsqueeze(torch.from_numpy(data), 0)
            print('data.shape', data.shape)
            pred_noise = model(X_valid)
            recon = torch.clamp(X_valid - pred_noise, 0., 1.)
            writer.image(
                f'extra/{k}_noisy:recon',
                make_grid([X_valid[0], recon[0]]),
                epoch
            )
        torch.save(model.state_dict(), os.path.join(out_loc, 'net.pth'))
