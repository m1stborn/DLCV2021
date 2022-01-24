# Problem 1: Few-shot learning - Prototypical Network 

A pytorch implementation of Prototypical Network. 
With ConvNet-4 backbone on miniImageNet.

### Usage

Training

```
python train_p1.py
```

Testing

```
python test_p1.py <--output_csv output.csv> <--ckpt your_ckpt.pt>
```

# Problem 2: Self-Supervised Pre-training for Image Classification

A pytorch implementation of BYOL base on [[lucidrains/byol-pytorch]](https://github.com/lucidrains/byol-pytorch)

Training

```
python train_p2.py
```

Testing

```
python test_p2.py <--output_csv output.csv> <--ckpt your_ckpt.pt>
```