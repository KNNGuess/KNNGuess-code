import torch
from model import make_model
from PW2SEQ import PW2SEQ
import numpy as np
import config
pw2seq=PW2SEQ()
from knn.datastore.datastore import Datastore
from knn.retriever.retriever import Retriever
from knn.combiner.combiner import Combiner
from lz_delete import beam_decode2
pw2seq_dict=pw2seq.pw2seq_dict
from utils import kb
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--source_password",type=str,default='Orange0301')
args=parser.parse_args()
source_pw=args.source_password

def translate2(src, model, use_beam=True):
    if config.use_datastore:
        datastore = Datastore.load(config.knn_datastore_path, load_list=["vals"])
        datastore.load_faiss_index("keys")
        retriever = Retriever(datastore=datastore, k=config.k)
        combiner = Combiner(lambda_=config.lambda_,
                        temperature=config.knn_temperature, probability_dim=config.src_vocab_size)
    else:
        datastore=0
        retriever=0
        combiner=0
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result = beam_decode2(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           datastore,retriever,combiner)
        translation = [(_s[0],pw2seq.inverse_transform(_s[1])) for _s in decode_result]
        decode_ans_list=[x[1] for x in translation]
        return decode_ans_list
    
if __name__=='__main__':
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    BOS = config.bos_idx 
    EOS = config.eos_idx 
    sent=source_pw
    src_tokens = [[BOS] + pw2seq.transform(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    trans=translate2(batch_input, model)
    ans=[kb.keyseq_to_word(x) for x in trans]
    print(ans)