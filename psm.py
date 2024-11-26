from PW2SEQ import PW2SEQ
import config
import numpy as np
from model import make_model
import torch
pw2seq=PW2SEQ()
pw2seq_dict=pw2seq.pw2seq_dict
from knn.datastore.datastore import Datastore
from knn.retriever.retriever import Retriever
from knn.combiner.combiner import Combiner
from utils import kb
from torch.autograd import Variable
from data_loader import subsequent_mask
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--source_password",type=str,default='Orange0301')
parser.add_argument("--target_password",type=str,default='orange1')
args=parser.parse_args()
source_pw=args.source_password
target_pw=args.target_password
datastore = Datastore.load(config.knn_datastore_path, load_list=["vals"])
datastore.load_faiss_index("keys")
retriever = Retriever(datastore=datastore, k=config.k)
combiner = Combiner(lambda_=config.lambda_,
                temperature=config.knn_temperature, probability_dim=config.src_vocab_size)



def beam_decode3(model,src,src_mask,target,pad,bos,eos,datastore,retriever,combiner):
    ans=[]

    src_enc=model.encode(src,src_mask) # batch_size , seq_len , hidden_size
    #print(src_enc.shape)
    begin=torch.ones(1, 1).fill_(bos).type_as(src.data)
    out=model.decode(src_enc,src_mask,Variable(begin),Variable(subsequent_mask(begin.size(1)).type_as(src.data)))
    prob=model.generator(out[:,-1])

    prob=torch.exp(prob)
    results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"])
    knn_prob = combiner.get_knn_prob(**results, device=config.device)
    word_prob=prob*config.lambda_+knn_prob.squeeze(1)*(1-config.lambda_)
    ans.append((word_prob.squeeze(0)[pw2seq_dict[target[0]]].item())/(torch.sum(word_prob.squeeze(0))).item())
    prob=torch.log(word_prob) # 1,54
    
    res=[[bos,pw2seq_dict[target[0]]]]
    res_tensor=torch.tensor(res).to(config.device)

    for i in range(1,len(target)):

        out=model.decode(src_enc,src_mask,Variable(res_tensor),Variable(subsequent_mask(res_tensor.size(1)).type_as(src.data)))
        next_prob=model.generator(out[:,-1])

        next_prob=torch.exp(next_prob)
        results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"]) # result['vals'].shape : beam_hold , k
        knn_prob = combiner.get_knn_prob(**results, device=config.device) # beam_hold , 1 ,vocab_size
        next_prob=next_prob*config.lambda_+knn_prob.squeeze(1)*(1-config.lambda_)
        ans.append((next_prob.squeeze(0)[pw2seq_dict[target[i]]].item())/(torch.sum(next_prob.squeeze(0))).item())
        prob=torch.log(word_prob)
        
        res[0]+=[pw2seq_dict[target[i]]]
        res_tensor=torch.tensor(res).to(config.device)
    return ans





if __name__=='__main__':
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                    config.d_model, config.d_ff, config.n_heads, config.dropout)
    with torch.no_grad():
        BOS = config.bos_idx  # 2
        EOS = config.eos_idx  # 3
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        sour_kb=kb.word_to_keyseq(source_pw)
        target_kb=kb.word_to_keyseq(target_pw)
        src_tokens = [[BOS] + pw2seq.transform(sour_kb) + [EOS]]
        src = torch.LongTensor(np.array(src_tokens)).to(config.device)
        src_mask = (src != 0).unsqueeze(-2)
        x=beam_decode3(model,src, src_mask, target_kb,
                            config.padding_idx, config.bos_idx, config.eos_idx,
                        datastore,retriever,combiner)
        print(x)

