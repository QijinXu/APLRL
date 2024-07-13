import os

import torch
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data, postprocess
from criteria.loss import generator_loss_func, discriminator_loss_func
import utils.utils_image as cal
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
def train(opts, image_data_loader, generator, discriminator, extractor,  generator_optim, discriminator_optim, is_cuda, test_opts, image_data_loader_test, lpips_criterion, aai_extractor):

    image_data_loader = sample_data(image_data_loader)
    image_data_loader_test = sample_data(image_data_loader_test)
    pbar = range(opts.train_iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)
    
    if opts.distributed:
        generator_module, discriminator_module = generator.module, discriminator.module
    else:
        generator_module, discriminator_module = generator, discriminator
    
    writer = SummaryWriter(opts.log_dir)
    
    
    
    for index in pbar:
        
        i = index + opts.start_iter
        if i > opts.train_iter:
            print('Done...')
            break

        ground_truth, mask, edge, gray_image = next(image_data_loader)

        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask

        # ---------
        # Generator
        # ---------
        requires_grad(generator, True)
        # for name, parameter in generator.named_parameters():
        #       if 'aai_extractor_1' in name:
        #           parameter.requires_grad = False   
        requires_grad(discriminator, False)

        output, projected_image, aai_gt, aai_out = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask, ground_truth, is_train=True)
        comp = ground_truth * mask + output * (1 - mask)
        with torch.no_grad():
            generate_dict = {k: v for k, v in generator_module.state_dict().items() if k in aai_extractor.state_dict()}
            aai_extractor.load_state_dict(generate_dict)
            aai_extractor.eval()
            aai_gt = aai_extractor(ground_truth,torch.ones_like(ground_truth))
        output_pred = discriminator(output, gray_image,  is_real=False)
        
        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)
        # ma = torch.ones_like(ground_truth)
        # _, _, rec_img_code = reconstructor(ground_truth, ma)
        
        generator_loss_dict = generator_loss_func(
            mask, output, ground_truth, output_pred, 
            vgg_comp, vgg_output, vgg_ground_truth, 
            # img_code, rec_img_code.detach()
            aai_gt, aai_out
        )

        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS + \
                         generator_loss_dict['loss_aai'] * opts.AAI_LOSS
        generator_loss_dict['loss_joint'] = generator_loss
        
        generator_optim.zero_grad()
        generator_loss.backward()
        generator_optim.step()

        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        real_pred= discriminator(ground_truth, gray_image, is_real=True)
        fake_pred= discriminator(output.detach(), gray_image, is_real=False)

        discriminator_loss_dict = discriminator_loss_func(real_pred, fake_pred, edge)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss

        discriminator_optim.zero_grad()
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)

        pbar_g_loss_hole = generator_loss_dict_reduced['loss_hole'].mean().item()
        pbar_g_loss_valid = generator_loss_dict_reduced['loss_valid'].mean().item()
        pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
        pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
        pbar_g_loss_adversarial = generator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_g_loss_intermediate = generator_loss_dict_reduced['loss_intermediate'].mean().item()
        pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()

        pbar_d_loss_adversarial = discriminator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()
        

        if get_rank() == 0:

            pbar.set_description((
                f'g_loss_joint: {pbar_g_loss_joint:.4f} '
                f'd_loss_joint: {pbar_d_loss_joint:.4f}'
            ))

            writer.add_scalar('g_loss_hole', pbar_g_loss_hole, i)
            writer.add_scalar('g_loss_valid', pbar_g_loss_valid, i)
            writer.add_scalar('g_loss_perceptual', pbar_g_loss_perceptual, i)
            writer.add_scalar('g_loss_style', pbar_g_loss_style, i)
            writer.add_scalar('g_loss_adversarial', pbar_g_loss_adversarial, i)
            writer.add_scalar('g_loss_intermediate', pbar_g_loss_intermediate, i)
            writer.add_scalar('g_loss_joint', pbar_g_loss_joint, i)

            writer.add_scalar('d_loss_adversarial', pbar_d_loss_adversarial, i)
            writer.add_scalar('d_loss_joint', pbar_d_loss_joint, i)

            if i % opts.save_interval == 0:
                
                
                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'discriminator': discriminator_module.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                 
                )

                

                print('start test...')
                with torch.no_grad():
                    generate_dict = {k: v for k, v in generator_module.state_dict().items() if k in aai_extractor.state_dict()}
                    aai_extractor.load_state_dict(generate_dict)
                    generator.eval()
                    aai_extractor.eval()
                    img_psnr = 0
                    img_ssim = 0
                    img_lpips = 0
                    
                    for _ in range(test_opts.number_eval):

                        ground_truth, mask, edge, gray_image = next(image_data_loader_test)
                        if is_cuda:
                            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

                        # input_aai = aai_extractor(ground_truth, torch.ones_like(ground_truth))
                        input_aai = aai_extractor(ground_truth * mask, mask.expand_as(ground_truth))
                        input_image, input_edge, input_gray_image, input_aai = ground_truth * mask, edge * mask, gray_image * mask, input_aai*mask
                        
                        output, __, __, __= generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask, input_aai, is_train=False)

                        
                        output_comp = ground_truth.clone() * mask + output.clone() * (1 - mask)
                        
                        output_comp = postprocess(output_comp)
                        
                        save_image(output_comp, test_opts.result_root + '/generate/' + '{:05d}.png'.format(_))
                        
                        # #save corrupt image
                        corrupt_img = postprocess(ground_truth * mask)
                        save_image(corrupt_img, test_opts.result_root + '/corrupt/' + '{:05d}.png'.format(_))
                        
                        # #save groundtruth
                        ground_truth = postprocess(ground_truth)
                        save_image(ground_truth, test_opts.result_root + '/ground_truth/' + '{:05d}.png'.format(_))
                        
                        # PSNR AND SSIM
                        output_comp=torch.squeeze(output_comp,dim=0).permute(1,2,0)
                        ground_truth=torch.squeeze(ground_truth,dim=0).permute(1,2,0)
                        psnr=cal.calculate_psnr(output_comp.cpu().detach().numpy(), ground_truth.cpu().detach().numpy())
                        ssim=cal.calculate_ssim(output_comp.cpu().detach().numpy(), ground_truth.cpu().detach().numpy())
                        lpips=calculate_lpips(Image.fromarray(output_comp.cpu().detach().numpy().astype(np.uint8)), Image.fromarray(ground_truth.cpu().detach().numpy().astype(np.uint8)), lpips_criterion)
                        # lpips= lpips_eval(output_comp, ground_truth)
                        img_psnr += psnr
                        img_ssim += ssim
                        img_lpips += lpips
                    img_psnr /= test_opts.number_eval
                    img_ssim /= test_opts.number_eval
                    img_lpips /= test_opts.number_eval
                    print('avg_psnr=%.2f'%img_psnr)
                    print('avg_ssim=%.3f'%img_ssim)
                    print('avg_lpips=%.3f'%img_lpips)
                with open("validation.txt", "a") as f:

                    # 将每一行转换为字符串，并使用空格分隔元素
                    f.write("avg_psnr={:.2f}".format(img_psnr) + " ")
                    f.write("avg_ssim={:.3f}".format(img_ssim) + " ")
                    f.write("avg_lpips={:.3f}".format(img_lpips) + "\n")

