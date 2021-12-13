# wget https://www.dropbox.com/s/ml2ibgde2rf405t/ViT-d3dbc53c.pt?dl=1 -O ViT-d3dbc53c.pt
wget https://www.dropbox.com/s/jmh00hjsre1oqjk/ViT-d231a067.pt?dl=1 -O ViT-d231a067.pt

python3 test_p1.py --p1_input_dir $1 --p1_output_file $2 --ckpt ViT-d231a067.pt

# local
# python test_p1.py  --p1_input_dir .\hw3_data\p1_data\val\ --p1_output_file .\result\p1\output.csv --ckpt .\ckpt\p1\ViT-d3dbc53c.pt