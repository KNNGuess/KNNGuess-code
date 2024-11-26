from data_loader import MTDataset
from torch.utils.data import DataLoader
import config
from model import make_model
import torch
from tqdm import tqdm
from utils import similar,get_segment,kb
from data_loader import subsequent_mask
from torch.autograd import Variable
from PW2SEQ import PW2SEQ
from knn.datastore.datastore import Datastore
from knn.common_utils.function import global_vars
pw2seq=PW2SEQ()
pw2seq_dict=pw2seq.pw2seq_dict


global_vars()["datastore"] = Datastore(path=config.knn_datastore_path)
datastore = global_vars()["datastore"]


def get_knn_train_data():
    f=open(config.train_data_path,"r",encoding='utf-8')
    if config.dataset_have_email==True:
        a,b=1,2
    else:
        a,b=0,1
    with open(config.knn_train_data_path,"w",encoding='utf-8') as f1:
        for line in f:
            lis=line.strip('\n').split('\t')
            pw1=lis[a]
            pw2=lis[b]
            f1.write(line)
            segment1=get_segment(pw1)
            segment2=get_segment(pw2)
            for seg1,detail1 in segment1:
                for seg2,detail2 in segment2:
                    if seg1!=seg2:
                        continue
                    if detail1==detail2:
                        continue
                    kb_detail1=kb.word_to_keyseq(detail1)
                    kb_detail2=kb.word_to_keyseq(detail2)
                    if similar(kb_detail1,kb_detail2)>0.4:
                        f1.write('_'+'\t'+detail1+'\t'+detail2+'\t'+'_'+'\n')          
    f1.close()

def gen_datastore():
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                    config.d_model, config.d_ff, config.n_heads, config.dropout)
    train_dataset = MTDataset(config.knn_train_data_path,mode='others')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.train_batch_size,
                                collate_fn=train_dataset.collate_fn)
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        for batch in tqdm(train_dataloader):
            batch_size, trg_seq_len = batch.trg.size()
            memory = model.encode(batch.src, batch.src_mask)
            tgt = torch.Tensor(batch_size, 1).fill_(config.bos_idx).type_as(batch.src.data)
            for s in range(1,trg_seq_len):
                tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(batch.src.data)
                out = model.decode(memory, batch.src_mask, Variable(tgt), Variable(tgt_mask)) # batch_size , seq_i_len , hidden_size
                last_hidden_state=out[:,-1] # batch_size , hidden_size
                keys=last_hidden_state
                values=batch.trg[:,s] # [batch_size]

                mask = values > 1
                keys = keys[mask]

                tgt = torch.cat((tgt, values.unsqueeze(1)), dim=1)
                values = values[mask]
                datastore['keys'].add(keys)
                datastore['vals'].add(values)
    datastore.dump()    # dump to disk
    datastore.build_faiss_index("keys", use_gpu=True)   # build faiss index

# first
get_knn_train_data()

# second
gen_datastore()
