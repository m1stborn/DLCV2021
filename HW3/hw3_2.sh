wget https://www.dropbox.com/s/y2ygzds3a6ou4lg/weight493084032.pth?dl=1 -O weight493084032.pth

python3 test_p2.py --path $1 --output $2 --ckpt weight493084032.pth

# local
# python test_p2.py --path  .\hw3_data\p2_data\images\ --output .\result\p2 --ckpt .\ckpt\p2\weight493084032.pth