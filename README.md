# Knee-MRI-Super-resolution
Easy training and test code for implementing MRI super-resolution. The model is based on [SA-INR](https://github.com/XinWang-99/SA-INR). A model trained on public knee dataset is also included in this repo, which can be directly used for inference.

## Dataset Class

Two dataset classes are defined in `make_dataset.py`:

### 1. `PairedDataset`
- **When to use**: Assumes low-resolution (LR) and high-resolution (HR) images exist as pairs.
- **Operation**: Loads a pair of images at a time and crops a patch of size 64x64x64 from the same location in both images.

### 2. `MakeDataset`
- **When to use**: Only HR images are available.
- **Operation**: Uses a predefined downsampling function to simulate the corresponding LR images from the HR images.

## Training

Training procedures are implemented in `train.py`.

**To run training**:
```bash
python train.py --gpu [gpu number] --save_dir [directory to save model] --gradient_loss [optional, defaults to L1 loss only] --gan_loss [optional] --preTrainPath [path to pretrained model]
```
- **--gpu**: Specifies the GPU number to use.
- **--save_dir**: Directory where the model files will be saved.
- **--gradient_loss and --gan_loss**: Optional parameters to specify the inclusion of gradient or GAN-based losses in addition to the default L1 loss.
- **--preTrainPath**: Specifies the path to a pre-trained model if continuing training from a checkpoint.

## Testing

Testing procedures are described in inference.py.

**To run testing**:

```bash
python inference.py --gpu [gpu number] --i [path of input .nii file] --o [path of output .nii file]
```
For thick-layer images above 1mm: The script normalizes layer thickness to 1mm.

- **--gpu**: Specifies the GPU number to use.
- **--i**: Input path for the low-resolution (thick-slice) .nii file to be processed.
- **--o**: Output path for the resulting high-resolution (thin-slice) .nii file.
