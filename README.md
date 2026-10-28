# DMG-Net

Official PyTorch implementation of the paper *"A Novel Directional and
Multi-Granular Network for Medical Image Segmentation"*.

DMG-Net is a hybrid U-shaped architecture that combines a Directional Shift
Attention Block (DSAB) in the encoder with a Multi-Granularity Context Fusion
(MGCF) module in the decoder, for accurate segmentation of anatomical
structures and lesions across CT, MRI, and endoscopic images.

## Pretrained models

The trained weights used in the paper are available via Baidu Netdisk:

> https://pan.baidu.com/s/1IHGYCRs4HZPJjyni__0sIw &nbsp;&nbsp;(code: `j8t7`)

| File | Dataset |
|---|---|
| `synapse.pth` | Synapse multi-organ CT |
| `acdc.pth` | ACDC cardiac MRI |
| `polyp.pth` | Polyp (endoscopy) |

The ImageNet-21k pretrained ViT backbone (`R50+ViT-B_16.npz`) follows the
TransUNet release; place it under `../model/vit_checkpoint/imagenet21k/`.

## Requirements

- Python 3.7+
- PyTorch >= 1.7
- medpy, SimpleITK, scipy, tqdm, opencv-python, pandas, matplotlib, seaborn

## Data preparation

Follow the TransUNet preprocessing to obtain the `npz`/`npy` volumes:

- **Synapse**: `../data/Synapse/` (train volumes + `test_vol_h5`), splits in `lists/lists_Synapse/`
- **ACDC**: `../data/ACDC/` with `train_slices.list`, `val.list`, `test.list`
- **Polyp**: `../data/polyp/TrainDataset/{images,masks}` and
  `../data/polyp/TestDataset/{CVC-ClinicDB,Kvasir,CVC-ColonDB,CVC-300}/{images,masks}`

## Training

```bash
python train_synapse.py   # Synapse, 9 classes, img_size 224
python train_acdc.py      # ACDC, 4 classes, img_size 224
python train_polyp.py     # Polyp, 1 class, img_size 352
```

All hyperparameters (batch size, learning rate, number of epochs, random seed
`1234`, backbone `R50-ViT-B_16`) are set as script defaults.

## Testing

```bash
python test_synapse.py
python test_acdc.py
python test_polyp.py
```

## Acknowledgements

This code builds on [TransUNet](https://github.com/Beckschen/TransUNet).

## License

This project is released under the MIT License. See `LICENSE`.
