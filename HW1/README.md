# Image Classification

A pytorch implementation of Vgg16 using pretrained model.

### Usage

* Training

  ```
  python train_p1.py <--epochs 20> <--batch_size 32> <--lr 0.001>
  ```
* Testing

  ```
  python3 test_p1.py <--p1_input_dir ./p1_data/val_50> <--p1_output_file output.csv> <--ckpt your_ckpt.pt>
  ```

### Model Performance


|  Model  | Accuracy |
| :--------: | :--------: |
|  Vgg16  |  0.7268  |
| Resnet50 |  0.8620  |

# Semantic Segmentation

A pytorch implementation of Vgg16FCN32s and Vgg16FCN8s

### Usage

* Training

  ```
  python train_p2.py <--epochs 20> <--batch_size 16> <--lr 0.0005>
  ```
* Testing

  ```
  python3 test_p2.py <--p2_input_dir ./p2_data/> <--p2_output_file ./predict> <--ckpt your_ckpt.pt>
  ```

### Model Performance


|    Model    | Mean IoU |
| :-----------: | :--------: |
| Vgg16FCN32s |  0.676  |
| Vgg16FCN8s |  0.696  |

### Prediction Visualization


| Validation image | 0010                                             | 0097                                             | 0107                                             |
| :----------------: | -------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
|    Satellite    | ![image](image/0010_sat.jpg) | ![image](image/0097_sat.jpg)                     | ![image](image/0107_sat.jpg)                     |
|     Epoch 1     | ![image](image/9cb47a9a-0010-epochs_0_mask.png)  | ![image](image/9cb47a9a-0097-epochs_0_mask.png)  | ![image](image/9cb47a9a-0107-epochs_0_mask.png)  |
|     Epoch 10     | ![image](image/9cb47a9a-0010-epochs_9_mask.png)  | ![image](image/9cb47a9a-0097-epochs_9_mask.png)  | ![image](image/9cb47a9a-0107-epochs_9_mask.png)  |
|     Epoch 20     | ![image](image/9cb47a9a-0010-epochs_19_mask.png) | ![image](image/9cb47a9a-0097-epochs_19_mask.png) | ![image](image/9cb47a9a-0107-epochs_19_mask.png) |
|     Epoch 27     | ![image](image/9cb47a9a-0010-epochs_26_mask.png) | ![image](image/9cb47a9a-0097-epochs_26_mask.png) | ![image](image/9cb47a9a-0107-epochs_26_mask.png) |
|   Ground Truth   | ![image](image/0010_mask.png)                    | ![image](image/0097_mask.png)                    | ![image](image/0107_mask.png)                    |
