import warnings
warnings.filterwarnings("ignore")

import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm
# from imageio import imsave

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.utils import save_image

from models.generator.generator import Generator, Extractor
from datasets.dataset import create_image_dataset
from options.test_options import TestOptions
from utils.misc import sample_data, postprocess
import utils.utils_image as cal
from PIL import Image
import torchvision.transforms as transforms
from lpips import LPIPS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
lpips_criterion = LPIPS(net='vgg').cuda()
# lpips_eval = LPIPS('vgg', version='0.1').cuda()
def calculate_lpips(img1, img2, lpips_criterion):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)


    img1 = torch.autograd.Variable(img1, requires_grad=False).cuda()
    img2 = torch.autograd.Variable(img2, requires_grad=False).cuda()


    lpips_value = lpips_criterion(img1, img2).item()

    return lpips_value

is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available')
    cudnn.enable = True
    cudnn.benchmark = True

opts = TestOptions().parse

os.makedirs('{:s}'.format(opts.result_root), exist_ok=True)
os.makedirs('{:s}'.format(opts.result_root+'/generate/'), exist_ok=True)
os.makedirs('{:s}'.format(opts.result_root+'/corrupt/'), exist_ok=True)
os.makedirs('{:s}'.format(opts.result_root+'/ground_truth/'), exist_ok=True)
# model & load model
generator = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
aai_extractor = Extractor()
if opts.pre_trained != '':
    generator.load_state_dict(torch.load(opts.pre_trained)['generator'])
else:
    print('Please provide pre-trained model!')

if is_cuda:
    generator, aai_extractor = generator.cuda(), aai_extractor.cuda()

# dataset
image_dataset = create_image_dataset(opts)
image_data_loader = data.DataLoader(
    image_dataset,
    batch_size=opts.batch_size,
    shuffle=False,
    num_workers=opts.num_workers,
    drop_last=False
)
image_data_loader = sample_data(image_data_loader)

print('start test...')
with torch.no_grad():

    generate_dict = {k: v for k, v in generator.state_dict().items() if k in aai_extractor.state_dict()}
    aai_extractor.load_state_dict(generate_dict)
    generator.eval()
    aai_extractor.eval()
    img_psnr = 0
    img_ssim = 0
    img_lpips = 0
    img_l1 = 0
    for _ in tqdm(range(opts.number_eval)):

        ground_truth, mask, edge, gray_image = next(image_data_loader)
        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()
        input_aai = aai_extractor(ground_truth*mask, mask.expand_as(ground_truth))
        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask
        output, __, __, dc_aai= generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask, input_aai, is_train=False)

        output_comp = ground_truth * mask + output * (1 - mask)
        
        output_comp = postprocess(output_comp)
        
        save_image(output_comp, opts.result_root + '/generate/' + '{:05d}.png'.format(_))
        # save_image(output_comp, opts.result_root  + '{:05d}.png'.format(_))
        # save corrupt image
        corrupt_img = postprocess(ground_truth * mask)
        save_image(corrupt_img, opts.result_root + '/corrupt/' + '{:05d}.png'.format(_))
        
        #save groundtruth
        ground_truth = postprocess(ground_truth)
        save_image(ground_truth, opts.result_root + '/ground_truth/' + '{:05d}.png'.format(_))
        
        # PSNR AND SSIM
        output_comp=torch.squeeze(output_comp,dim=0).permute(1,2,0)
        ground_truth=torch.squeeze(ground_truth,dim=0).permute(1,2,0)
        psnr=cal.calculate_psnr(output_comp.cpu().numpy(), ground_truth.cpu().numpy())
        ssim=cal.calculate_ssim(output_comp.cpu().numpy(), ground_truth.cpu().numpy())
        lpips=calculate_lpips(Image.fromarray(output_comp.cpu().numpy().astype(np.uint8)), Image.fromarray(ground_truth.cpu().numpy().astype(np.uint8)), lpips_criterion)
        # lpips= lpips_eval(output_comp, ground_truth)
        img_psnr += psnr
        img_ssim += ssim
        img_lpips += lpips
    img_psnr /= opts.number_eval
    img_ssim /= opts.number_eval
    img_lpips /= opts.number_eval
    print('avg_psnr=%.2f'%img_psnr)
    print('avg_ssim=%.3f'%img_ssim)
    print('avg_lpips=%.3f'%img_lpips)
    with open("test_result.txt", "a") as f:

        # 将每一行转换为字符串，并使用空格分隔元素
        f.write("avg_psnr={:.2f}".format(img_psnr) + "\n")
        f.write("avg_ssim={:.3f}".format(img_ssim) + "\n")
        f.write("avg_lpips={:.3f}".format(img_lpips) + "\n")
        
        
        
        
        
        
        