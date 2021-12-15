# Problem 1: Image Classification with Vision Transformer (ViT)

A pytorch implementation of ViT base on [[lukemelas/PyTorch-Pretrained-ViT]](https://github.com/lukemelas/PyTorch-Pretrained-ViT)

### Usage

Training

```
python train_p1.py <--epochs 10> <--batch_size 8> <--lr 0.001>
```

Testing

```
python test_p1.py <--p1_input_dir folder_to_valid> <--p1_output_file output.csv> <--ckpt your_ckpt.pt>
```

### Model Performance

|       Model       | Accuracy |
|-------------------|:--------:|
| “B_16_imagenet1k” |  0.9526  |

### Position Embedding Visualization

<img src="https://raw.githubusercontent.com/m1stborn/DLCV2021/master/HW3/image/d231a067_position_embedding.png"  alt="">

### Attention Map Visualization 

<img src="https://raw.githubusercontent.com/m1stborn/DLCV2021/master/HW3/image/d231a067-31_4838_l11.png"  alt="" width="60%" height="60%">
 
# Problem 2:  Visualization in Image Captioning (Transformer-based)

A pytorch implementation of Visualization of CAption TRansformer base on [[saahiluppal/catr]](https://github.com/saahiluppal/catr)

### Attention Map Visualization 

The attention map corresponding to series of predicted caption.

<img alt="" src="https://raw.githubusercontent.com/m1stborn/DLCV2021/master/HW3/image/bike.png">