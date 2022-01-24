wget https://www.dropbox.com/s/75tfvbkmm01p8do/Resnet-da022668.pt?dl=1  -O Resnet-da022668.pt

python3 test_p2.py --test_csv $1 --test_data_dir $2 --output_csv $3 --ckpt Resnet-da022668.pt

# Local run:
#python3 test_p2.py --ckpt .\ckpt\best\Resnet-da022668.pt
