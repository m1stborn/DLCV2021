# Problem 1: General Adversarial Network (GAN)

A pytorch implementation of GAN

### Usage

Training

```
python train_p1.py <--epochs 10> <--batch_size 8> <--lr 0.001>
```

Testing

```
python test_p1.py  <--p1_output_dir output_folder> <--ckpt your_ckpt.pt>
```

### Model Performance

|             Metric              |  score  |
|:-------------------------------:|:-------:|
| FID(Fréchet inception distance) | 26.2067 |
|      IS (Inception score)       | 2.1085  |


### Using first 32 latent noise generated images

<img src="https://raw.githubusercontent.com/m1stborn/DLCV2021/master/HW2/assets/images/First_32.png"  alt="">

# problem 2: Auxiliary Classifier GAN (ACGAN)

### Usage

Training

```
python train_p2.py <--epochs 10> <--batch_size 8> <--lr 0.001>
```

Testing

```
python test_p2.py  <--p2_output_dir output_folder> <--ckpt your_ckpt.pt>
```
### Using first 10 latent noise of each digit(from 0 to 9) generated images

<img src="https://raw.githubusercontent.com/m1stborn/DLCV2021/master/HW2/assets/images/First_10.png"  alt="">

# problem 3: Domain-Adversarial Neural Network (DANN)

### Model Performance

|   Domain    | MNIST-M → USPS | SVHN → MNIST-M | USPS → SVHN |
|:-----------:|:--------------:|:--------------:|:-----------:|
| Source only |     0.6393     |     0.4604     |   0.2219    |
|    DANN     |     0.8241     |     0.5078     |   0.3421    |
| Target only |     0.9606     |     0.9791     |   0.9174    |
