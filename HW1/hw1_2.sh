wget https://www.dropbox.com/s/33l6dwkbsefseod/Vgg16FCN8-9cb47a9a.pt?dl=1

python3 test_p2.py --p2_input_dir $1 --p2_output_dir $2 --ckpt Vgg16FCN8-9cb47a9a.pt