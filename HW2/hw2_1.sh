# TODO: wget
#wget https://www.dropbox.com/s/49eor89c2ydu9hx/Resnet-1acf9fda.pt?dl=1 -O Resnet-1acf9fda.pt
# TODO: rename ckpt
python3 test_p1.py --ckpt .\ckpt\DCGAN-cc02f2b1.pt --p1_output_dir $1

#FID score
#python -m pytorch_fid .\p1_result\1000_generated_images .\hw2_data\face\test\ --num-workers 2
