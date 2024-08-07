import warnings
warnings.filterwarnings("ignore")

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data

from models.discriminator.discriminator import Discriminator
from models.generator.generator import Generator, Extractor

from models.generator.vgg16 import VGG16FeatureExtractor
from options.train_options import TrainOptions
from datasets.dataset import create_image_dataset
from utils.distributed import synchronize
from utils.ddp import data_sampler
from trainer import train
from models.reconstructor.reconstructor import Reconstructor


from options.test_options import TestOptions
import utils.utils_image as cal
from PIL import Image
import torchvision.transforms as transforms
from lpips import LPIPS
opts = TrainOptions().parse

os.makedirs('{:s}'.format(opts.save_dir), exist_ok=True)

is_cuda = torch.cuda.is_available()
if is_cuda:
    
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('GPU number: ', n_gpu)
    opts.distributed = n_gpu > 1
    if opts.distributed:
        torch.cuda.set_device(opts.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

# model & load model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
discriminator = Discriminator(image_in_channels=3, edge_in_channels=2)
extractor = VGG16FeatureExtractor()
aai_extractor = Extractor()

# reconstructor = Reconstructor()
# reconstructor.load_state_dict(torch.load('snapshots/ckpt_celeba/reconstructor_100000.pt')['reconstructor'])
# reconstructor.train(False)
# cuda
if is_cuda:
    generator, discriminator, extractor, aai_extractor = generator.cuda(), discriminator.cuda(), extractor.cuda(), aai_extractor.cuda()


    

# optimizer
if opts.finetune == True:
    print('Fine tune...')
    lr = opts.lr_finetune
    generator.freeze_ec_bn = True
else:
    lr = opts.gen_lr

generator_optim = optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr * opts.D2G_lr)

# load checkpoints
# generator.load_state_dict(torch.load('snapshots/Paris StreetView/290000.pt')['generator'])
if opts.pre_trained != '':
    ckpt_dict = torch.load(opts.pre_trained, map_location=lambda storage, loc: storage)
    opts.start_iter = ckpt_dict['n_iter']
    generator.load_state_dict(ckpt_dict['generator'])
    discriminator.load_state_dict(ckpt_dict['discriminator'])

    print('Starting from iter', opts.start_iter)
else:
    print('Starting from iter', opts.start_iter)
# freeze aai_extractor in generator
# for name, parameter in generator.named_parameters():
#       if 'aai_extractor_1' in name:
#           parameter.requires_grad = False    

if opts.distributed:

    generator = nn.parallel.DistributedDataParallel(
        generator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator, 
        device_ids=[opts.local_rank],
        output_device=opts.local_rank,
        broadcast_buffers=False,
    )

# dataset
image_dataset = create_image_dataset(opts)
print(image_dataset.__len__())

image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    sampler=data_sampler(
        image_dataset, shuffle=True, distributed=opts.distributed
    ),
    drop_last=True
)

#validation
lpips_criterion = LPIPS(net='vgg').cuda()
test_opts = TestOptions().parse 
os.makedirs(test_opts.result_root + '/ground_truth/', exist_ok=True)
os.makedirs(test_opts.result_root + '/generate/', exist_ok=True)
os.makedirs(test_opts.result_root + '/corrupt/', exist_ok=True)
image_dataset_test = create_image_dataset(test_opts)
image_data_loader_test = data.DataLoader(
    image_dataset_test,
    batch_size=test_opts.batch_size,
    shuffle=False,
    num_workers=test_opts.num_workers,
    drop_last=False
)

# training
train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda, test_opts, image_data_loader_test, lpips_criterion, aai_extractor)



