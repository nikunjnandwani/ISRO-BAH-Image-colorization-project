# Infrared Image Colorization & Enhancement Pipeline

This repository contains an end-to-end framework for super-resolving and colorizing multi-band infrared (IR) satellite imagery from Landsat 8/9.

## Architecture

The pipeline follows this flow:
1. **Input**: 4-channel IR stack (B5 NIR, B6 SWIR1, B7 SWIR2, B10 Thermal).
2. **Super-Resolution**: B10 is super-resolved from its native ~100m equivalent to 30m using a lightweight ESRGAN (RRDB + VGG discriminator).
3. **Colorization**: The stacked 30m bands are fed into a U-Net generator (with self-attention and spectral initialization) to predict a 3-channel RGB image.
4. **Semantic Constraint**: During training, a semantic branch computes NDVI/NDWI indices to ensure physical constraints (e.g., water maps to blue, vegetation to green).
5. **Discriminator**: A PatchGAN discriminator evaluates the realism of the output.

## Setup

1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your raw Landsat GeoTIFF files into the `archive/` directory.

## Training on Google Colab (Recommended)

Training this pipeline on a CPU or a low-VRAM GPU (like a 6GB RTX 3050) is very slow. It is highly recommended to use Google Colab's free T4 GPU.

1. Zip the entire project folder and upload it to your Google Drive, or upload it unzipped to a specific folder (e.g., `My Drive/ISRO_BAH`).
2. Open the included `colab_training.ipynb` notebook in Google Colab.
3. The notebook will guide you through:
   - Mounting Google Drive
   - Installing requirements
   - Running `prepare_dataset.py` (which takes the TIFFs and extracts 128x128 patches)
   - Training the ESRGAN model (`train_sr.py`)
   - Training the Pix2Pix colorization model (`train_colorize.py`)

## Local Execution

If you wish to run the pipeline locally:

**1. Data Preparation**
```bash
python data/prepare_dataset.py --archive_dir archive --output_dir prepared_data
```
This extracts overlapping patches, handles masking (NaN values), and computes normalization stats.

**2. Train Super-Resolution (ESRGAN)**
```bash
python train_sr.py --data_dir prepared_data --checkpoint_dir checkpoints/sr --epochs_pretrain 50 --epochs_gan 50
```

**3. Train Colorization (Pix2Pix)**
```bash
python train_colorize.py --data_dir prepared_data --checkpoint_dir checkpoints/colorize --epochs 200
```

**4. Inference**
Run the trained models on a full GeoTIFF tile:
```bash
python inference.py --input archive/landsat_sundarbans.tif \
                    --sr_checkpoint checkpoints/sr/sr_final.pth \
                    --colorize_checkpoint checkpoints/colorize/colorize_best.pth \
                    --output outputs/sundarbans_colorized.tif
```

**5. Evaluation**
```bash
python evaluate.py --data_dir prepared_data/test \
                   --checkpoint checkpoints/colorize/colorize_best.pth \
                   --output_dir outputs/evaluation
```
