import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import lpips
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='experiments/basic_sr_ffhq_210809_142238/results')
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.path)))
    real_names.sort()
    fake_names.sort()
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # best forward scores

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    avg_lpips_vgg = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = rname.rsplit("_hr")[0]
        fidx = fname.rsplit("_sr")[0]
        #assert ridx == fidx, 'Image ridx:{ridx}!=fidx:{fidx}'.format(
        #    ridx, fidx)
        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        sr_img = torch.tensor(sr_img).permute(2, 0, 1).unsqueeze(0)
        hr_img = torch.tensor(hr_img).permute(2, 0, 1).unsqueeze(0)
        lpips = loss_fn_alex(sr_img, hr_img).item()
        lpips_vgg = loss_fn_vgg(sr_img, hr_img).item()
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips
        avg_lpips_vgg += lpips_vgg
        if idx % 1 == 0:
            print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}, LPIPS:{:.4f}, LPIPS-VGG:{:.4f}'.format(idx, psnr, ssim, lpips, lpips_vgg))

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    avg_lpips = avg_lpips / idx
    avg_lpips_vgg = avg_lpips_vgg / idx

    # log
    print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    print('# Validation # LPIPS: {:.4e}'.format(avg_lpips))
    print('# Validation # LPIPS-VGG: {:.4e}'.format(avg_lpips_vgg))
