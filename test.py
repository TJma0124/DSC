import os
import sys
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model import *

from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("SDCE")
parser.add_argument('--data_path', type=str, default='./datasets/UIEB-S',#UIEB-S/MIT
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results/UIEB-S', help='location of the data corpus')#UIEB-S/MIT
# parser.add_argument('--model', type=str, default='./weights/weights_UnderwaterReal.pt', help='location of the data corpus')
# parser.add_argument('--model', type=str, default='./weights/weights_UnderwaterSny.pt', help='location of the data corpus')
parser.add_argument('--model', type=str, default='./weights/weights_Lowlight.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Network()
    pretrained_dict = torch.load(args.model)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    
    model.train()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            enh= model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            
            u_path = save_path + '/' + u_name
            save_images(enh, u_path)



if __name__ == '__main__':
    main()
