# Waving Goodbye to Low-Res: A Diffusion-Wavelet Approach for Image Super-Resolution
This work presents a novel Diffusion-Wavelet (DiWa) approach for Single-Image Super-Resolution (SISR). It leverages the strengths of Denoising Diffusion Probabilistic Models (DDPMs) and Discrete Wavelet Transformation (DWT). By enabling DDPMs to operate in the DWT domain, our DDPM models effectively hallucinate high-frequency information for super-resolved images on the wavelet spectrum, resulting in high-quality and detailed reconstructions in image space. 

## Brief

This is the official implementation of **Waving Goodbye to Low-Res: A Diffusion-Wavelet Approach for Image Super-Resolution** ([arXiv paper](https://arxiv.org/abs/2304.01994)) in **PyTorch**.
The repo was cleaned before uploading. Please report any bug.
It complements the inofficial implementation of **SR3** ([GitHub](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)).


## Usage

### Environment

```python
pip install -r requirement.txt
```

### Continue Training

```python
# Download the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```

### Data Preparation

If you don't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset) | [FFHQ 512×512](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256) | [CelebaMask-HQ 1024×1024](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Download the dataset and prepare it in **LMDB** (not DIV2K) or **PNG** format using script.
For DIV2K, remove the "-l" parameter and also use the preprocessing step described last in this section (to extract sub-images).

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

For DIV2K, you will need to extract the sub-images beforehand:
```python
python data/prepare_div2k.py  --path [dataset root]  --out [output root]
```
Note: LMDB does not work for DIV2K.

For the test datasets:
- [Set5](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
- [Set14](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
- [BSDS100](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)
- [DIV2K Validation](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

you need to put the files into the dataset folder and run
```python
python data/prepare_natural_tests.py
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

#### Configurations for Training


| Tasks                             | Config File                                              | 
|-----------------------------------|----------------------------------------------------------|
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [config/sr_wave_16_128.json](config/sr_wave_16_128.json) |  
| 64×64 -> 512×512 on FFHQ-CelebaHQ | [config/sr_wave_64_512.json](config/sr_wave_64_512.json) |   
| 48×48 -> 192×192 on DIV2K         | [config/sr_wave_48_192.json](config/sr_wave_48_192.json) |
| Ablation - baseline               | [config/sr_wave_48_192_abl_baseline.json](config/sr_wave_48_192_abl_baseline.json) |
| Ablation - Init. Pred. only       | [config/sr_wave_48_192_abl_pred_only.json](config/sr_wave_48_192_abl_pred_only.json) |
| Ablation - DWT only               | [config/sr_wave_48_192_abl_wave_only.json](config/sr_wave_48_192_abl_wave_only.json) |
| Ablation - DiWa                   | [config/sr_wave_48_192_abl_wave+pred.json](config/sr_wave_48_192_abl_wave+pred.json) |

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR/LPIPS metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the image path, then run the script:

```python
# run the script
python infer.py -c [config file]
```
