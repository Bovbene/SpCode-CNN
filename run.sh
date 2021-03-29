source activate TensorFlow
python preprocess.py -NI 10 -TD ./DATABASE/
python train.py --which_model SpCode-VDSR
python test.py --which_model SpCode-VDSR --up_scale 4 --file_name './DATABASE/Set5/woman_GT.bmp'