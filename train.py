import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from model import *
from multi_read_data import MemoryFriendlyLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("SDCE")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='3', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=400, help='epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
test_content = input("请输入实验内容：")
args.save = args.save + '/' + 'Train-'+ test_content +':{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'jpeg')
def trans_imagesnew(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)


    model = Network()

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    
    train_low_data_names = './datasets/UIEB-S/train-sny600'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = './datasets/UIEB-S/valid'
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    valid_low_data_names = '/data/matianjiao/shareG/MTJ/dataset/underwater/UIEB/test-sny'
    validDataset = MemoryFriendlyLoader(img_dir=valid_low_data_names, task='train')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=False,generator=torch.Generator(device = 'cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False,generator=torch.Generator(device = 'cuda'))
    valid_queue = torch.utils.data.DataLoader(
        validDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False,generator=torch.Generator(device = 'cuda'))

    total_step = 0
    writer = SummaryWriter(log_dir=args.save, flush_secs=30)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        Mylosses = []
        SSIMlosses = []
        Gradientlosses= []

        
        
        for batch_idx, (input, label, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            label = Variable(label, requires_grad=False).cuda()

            optimizer.zero_grad()
            loss,My_Loss, SSIM_Loss, gradientloss= model._loss(input, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            Mylosses.append(My_Loss.item())
            SSIMlosses.append(SSIM_Loss.item())
            Gradientlosses.append(gradientloss.item())
            logging.info('train-epoch %03d %03d %f |My: %f|SSIM: %f|Gradient: %f|', epoch, batch_idx, loss, My_Loss, SSIM_Loss,gradientloss)



        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))
        writer.add_scalar('average loss',np.average(losses), epoch)
        writer.add_scalar('average My loss', np.average(Mylosses), epoch)
        writer.add_scalar('average SSIM loss', np.average(SSIMlosses), epoch)
        writer.add_scalar('average Gradient loss', np.average(Gradientlosses), epoch)
        

        if True:
            model.train()
            with torch.no_grad():
                if epoch % 5 == 0 and total_step != 0:
                    for _, (input, image_name) in enumerate(test_queue):
                        input = Variable(input, volatile=True).cuda()
                        image_name = image_name[0].split('/')[-1].split('.')[0]
                        enh = model(input)
                        u_name = '%s.jpg' % (image_name + '_' + str(epoch))
                        u_path = image_path + '/' + u_name
                        save_images(enh, u_path)
                        


if __name__ == '__main__':
    main()
