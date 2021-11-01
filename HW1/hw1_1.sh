wget https://www.dropbox.com/s/49eor89c2ydu9hx/Resnet-1acf9fda.pt?dl=1 -O Resnet-1acf9fda.pt

python3 test_p1.py --p1_input_dir $1 --p1_output_file $2 --ckpt Resnet-1acf9fda.pt