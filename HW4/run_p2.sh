#python train_resnet_p2.py --no_pretrained --no_freeze # A
#python train_resnet_p2.py --ckpt .\ckpt\p2\pretrain_model_SL.pt --no_freeze --pretrained_sl # B
#python train_resnet_p2.py  --ckpt .\ckpt\p2\BYOL-088a1261.pt --no_freeze # C
python train_resnet_p2.py --ckpt .\ckpt\p2\pretrain_model_SL.pt --pretrained_sl # D
#python train_resnet_p2.py  --ckpt .\ckpt\p2\BYOL-088a1261.pt  # E

# A: 0.2414
# B: 0.3571
# C:
# D: 0.3399
# E: 0.3966
