import torch

# 1. check train_data_path
# 2. check guess_ans_path、guess_ans_path_nomix、test_data_path
# 3. check min_len
# 4. check language

#model>>>>>>>>>>>>>>>>>>
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.2
padding_idx = 1
bos_idx = 2
eos_idx = 3
src_vocab_size = 54
tgt_vocab_size = 54
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True
#<<<<<<<<<<<<<<<<<<<<<<


#train>>>>>>>>>>>>>>>>
epoch_num = 30
language='cn'
lr = 5e-4
sort_dataset=True
train_batch_size = 256
train_data_path = './tianya_dodonew_train.txt'
#<<<<<<<<<<<<<<<<<<<<<

#knn>>>>>>>>>>>>>>>>>>>
use_datastore=True
knn_temperature=300
k=32 
# NMT占比
lambda_=0.5
# knn占比
knn_lambda_=0.4
# local_knn占比
local_knn_lambda_=0.1
knn_datastore_path="./datastore"
knn_train_data_path=train_data_path[:-4]+"_knn.txt"
#<<<<<<<<<<<<<<<<<<<<<


dataset_have_email=True

#test>>>>>>>>>>>>>>>>>
TopK=1000
min_len=6
max_len = 20
# beam size for bleu
beam_width = 40
beam_hold=1000

model_path = './experiment/model.pth'
log_path = './experiment/train.log'
#<<<<<<<<<<<<<<<<<<<<<

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
