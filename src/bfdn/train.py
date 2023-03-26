import os
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from bfdn.models import DnCNN
from bfdn.data import Data


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data \
            .normal_(mean=0, std=math.sqrt(2. / 9. / 64.)) \
            .clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)
    return


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, ...], Img[i, ...],
                                        data_range=data_range)
    return (PSNR/Img.shape[0])


def main(batch_size, num_layers, learning_rate, num_epochs, milestone, seed,
         use_bias, out_loc, debug=False):
    torch.manual_seed(0)
    rng = np.random.default_rng(seed)
    if not os.path.isdir(out_loc):
        print('making directory:', out_loc)
        os.mkdir(out_loc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        train_set = Data(train=True, max_keys=20 * batch_size)
    else:
        train_set = Data(train=True)
    print('# of training samples:', len(train_set))
    valid_set = Data(train=False)
    train_set = DataLoader(dataset=train_set,
                           num_workers=4,
                           batch_size=batch_size,
                           shuffle=True)
    net = DnCNN(channels=1, num_of_layers=num_layers, bias=use_bias)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()
    criterion.to(device)
    model = nn.DataParallel(net, device_ids=[0]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(out_loc)
    noise_range = [0, 55]
    for epoch in range(num_epochs):
        if epoch == milestone:
            print('epoch:', epoch, 'learning rate', learning_rate / 10.0)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate / 10.0
        model.train()
        for i, data in enumerate(train_set):
            model.zero_grad()
            optimizer.zero_grad()
            data.to(device)
            shape = data.shape
            noise = torch.zeros(shape)
            stds = rng.uniform(*noise_range, size=shape[0])
            for j, s in enumerate(stds):
                noise[i, ...] = torch.FloatTensor(shape[1:])\
                    .normal_(mean=0, std=s/255.)
            noise.to(device)
            X_train = (data + noise).to(device)
            pred_noise = model(X_train)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            # results
            train_recon = torch.clamp(X_train - pred_noise, 0., 1.)
            psnr_train = batch_PSNR(train_recon, data, 1.)
            print("[epoch %d][%2d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch+1, i+1, len(train_set), loss.item(), psnr_train))
            if i % 10 == 0:
                writer.add_scalar('train loss', loss.item(), i)
                writer.add_scalar('PSNR on training data', psnr_train, i)
        model.eval()
        writer.add_image(
            'train clean image',
            make_grid(X_train, nrow=8, normalize=True, scale_each=True),
            epoch
        )
        writer.add_image(
            'train noisy image',
            make_grid(data, nrow=8, normalize=True, scale_each=True),
            epoch
        )
        writer.add_image(
            'train reconstructed image',
            make_grid(train_recon, nrow=8, normalize=True, scale_each=True),
            epoch
        )
        # validate
        psnr_val = 0
        loss = 0
        for k, data in enumerate(valid_set):
            data = torch.unsqueeze(data, 0)
            noise = torch.FloatTensor(data.size()).normal_(mean=0, std=25.0/255.)
            X_valid = (data + noise).to(device)
            pred_noise = torch.clamp(data - model(X_valid), 0., 1.)
            psnr_val += batch_PSNR(pred_noise, X_valid, 1.)
            loss += criterion(pred_noise, noise).item()
        psnr_val /= len(valid_set)
        loss /= len(valid_set)
        print("[epoch %d] valid loss: %.4f PSNR_val: %.4f" % (epoch+1, loss, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        torch.save(model.state_dict(), os.path.join(out_loc, 'net.pth'))


if __name__ == '__main__':
    from bfdn.etl import DATA_PATH
    main(50, 3, 0.01, 20, 10, 42, True, f'{DATA_PATH}/results', debug=True)
