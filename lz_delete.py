import operator
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from data_loader import subsequent_mask
import config
from PW2SEQ import PW2SEQ
from collections import defaultdict
from utils import kb,get_segment
import numpy as np
pw2seq=PW2SEQ()
pw2seq_dict=pw2seq.pw2seq_dict
ss=defaultdict(int)
total_chars='`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./ '
digits_set=set(y for y in [x for x in range(5,15)])
letters_set=set(y for y in [x for x in range(17,27)]+[x for x in range(30,39)]+[x for x in range(41,48)])


def get_local_res(new_res):
    local_res=[]
    for new_res_i in new_res:
        kb_new_res_str=pw2seq.inverse_transform(new_res_i)
        new_res_str=kb.keyseq_to_word(kb_new_res_str)
        segment=get_segment(new_res_str)
        #print(segment)
        if len(segment)==0:
            local_res.append([config.bos_idx,config.eos_idx])
            continue
        seg=segment[-1][1]
        seg_tokens=[config.bos_idx]+pw2seq.transform(seg)+[config.eos_idx]
        local_res.append(seg_tokens)
    #print(local_res)
    return local_res,pad_sequence([torch.LongTensor(np.array(l_)) for l_ in local_res],batch_first=True, padding_value=config.padding_idx).to(config.device)

    # another method
    # def get_flag(new_res_i):
    #     if new_res_i in digits_set:return "D"
    #     elif new_res_i in letters_set:return "L"
    #     else:return "S"
    # flag=get_flag(new_res[-1])
    # for i in range(len(new_res)-2,0,-1): 
    #     cur_flag=get_flag(new_res[i])
    #     if cur_flag!=flag:
    #         return [config.bos_idx]+new_res[i+1:]
    # return new_res
def get_dec_seq(local_res_nopadding):
    max_len=max([len(x) for x in local_res_nopadding])-1 # - EOS_TOKEN
    for i in range(len(local_res_nopadding)):
        num=max_len-len(local_res_nopadding[i])+1
        local_res_nopadding[i]=[config.padding_idx]*num+local_res_nopadding[i][:-1]
    return torch.tensor(local_res_nopadding).to(config.device)


def check(pw):
    pw=pw2seq.inverse_transform(pw)
    flag1=0
    flag2=0
    for c in pw:
        if c.isalpha():flag1=1
        if c.isdigit():flag2=1
    return flag1==1 and flag2==1

def beam_decode2(model,src,src_mask,max_len,pad,bos,eos,datastore,retriever,combiner):
    beam_width = config.beam_width
    beam_hold=config.beam_hold
    topk = config.TopK
    res=[] 
    decode_ans=[]
    src_enc=model.encode(src,src_mask) # batch_size , seq_len , hidden_size
    begin=torch.ones(1, 1).fill_(bos).type_as(src.data)
    out=model.decode(src_enc,src_mask,Variable(begin),Variable(subsequent_mask(begin.size(1)).type_as(src.data)))
    prob=model.generator(out[:,-1])
    prob=torch.exp(prob)
    results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"])
    knn_prob = combiner.get_knn_prob(**results, device=config.device)
    word_prob=prob*config.lambda_+knn_prob.squeeze(1)*(1-config.lambda_)
    prob=torch.log(word_prob)
    log_prob, indexes = torch.topk(prob, beam_width) # 1 * beam_width
    prob=[]
    for i in range(beam_width):
        p=log_prob[0][i].item()
        seq=indexes[0][i].item()
        res.append([bos,seq])
        prob.append(p)


    res_tensor=torch.tensor(res).to(config.device) # beam_width , 2
    prob=torch.tensor(prob).type_as(src.data).to(config.device) # beam_width
    new_src_enc=src_enc.repeat(1, beam_width, 1).view(beam_width, src_enc.size(1), src_enc.size(2))
    new_src_mask = src_mask.repeat(1, beam_width, 1).view(beam_width, 1, src_mask.shape[-1])
    out=model.decode(new_src_enc,new_src_mask,Variable(res_tensor),Variable(subsequent_mask(res_tensor.size(1)).type_as(src.data)))
    next_prob=model.generator(out[:,-1]) # beam_width , 98


    next_prob=torch.exp(next_prob)
    results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"]) # result['vals'].shape : beam_hold , k
    knn_prob = combiner.get_knn_prob(**results, device=config.device) # beam_hold , 1 ,vocab_size
    next_prob=next_prob*config.lambda_+knn_prob.squeeze(1)*(1-config.lambda_)
    next_prob=torch.log(next_prob)
    prob=prob.unsqueeze(1).expand_as(next_prob) # beam_width, 98
    next_prob=next_prob+prob
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if 2<config.min_len:
        next_prob[:,config.eos_idx]=torch.full((next_prob.size(0),),float('-inf'))
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    next_prob=next_prob.view(-1) # beam_width*98
    log_prob, indexes = torch.topk(next_prob, beam_hold) # [beam_hold]
    new_res=[]
    new_log_prob=[]
    for i in range(beam_hold):
        x=indexes[i].item()
        num=x//config.src_vocab_size
        last=x%config.src_vocab_size
        y=res[num]+[last]
        #res[num].append(last)
        if last==eos:
            if len(y)>config.min_len:
                if check(y): ########mod
                    decode_ans.append((log_prob[i].item(),y))
            continue
        new_res.append(y)
        new_log_prob.append(log_prob[i].item())

    for i in range(3,config.max_len):
        res_tensor=torch.tensor(new_res).to(config.device) # beam_hold, 3
        ans=new_res
        log_prob_tensor=torch.tensor(new_log_prob).to(config.device)
        #print(res_tensor.shape)
        new_src_enc=src_enc.repeat(1, res_tensor.size(0), 1).view(res_tensor.size(0), src_enc.size(1), src_enc.size(2)) # beam_hold, seq_len, 512
        #print(new_src_enc.shape)
        new_src_mask = src_mask.repeat(1, res_tensor.size(0), 1).view(res_tensor.size(0), 1, src_mask.shape[-1]) # beam_hold, 1, seq_len
        #print(new_src_mask.shape)
        assert new_src_enc.shape[0] == res_tensor.shape[0] == new_src_mask.shape[0]
        out=model.decode(new_src_enc,new_src_mask,Variable(res_tensor),Variable(subsequent_mask(res_tensor.size(1)).type_as(src.data)))
        next_prob=model.generator(out[:,-1]) # beam_hold , 98

        next_prob=torch.exp(next_prob)
        results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"])
        knn_prob = combiner.get_knn_prob(**results, device=config.device)
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        local_res_nopadding,local_res=get_local_res(new_res)
        local_src_mask = (local_res != pad).unsqueeze(-2)
        local_src_enc=model.encode(local_res,local_src_mask)
        dec_seq=get_dec_seq(local_res_nopadding)
        assert local_src_enc.shape[0] == dec_seq.shape[0] == local_src_mask.shape[0]
        out=model.decode(local_src_enc,local_src_mask,dec_seq,subsequent_mask(dec_seq.size(1)).type_as(src.data))
        results=retriever.retrieve(out[:,-1], return_list=["vals", "distances"])
        local_knn_prob=combiner.get_knn_prob(**results, device=config.device)
        next_prob=next_prob*config.lambda_+knn_prob.squeeze(1)*config.knn_lambda_+local_knn_prob.squeeze(1)*config.local_knn_lambda_
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #next_prob=next_prob*config.lambda_+knn_prob.squeeze(1)*(1-config.lambda_)  #  modify
        next_prob=torch.log(next_prob)
        log_prob_tensor=log_prob_tensor.unsqueeze(1).expand_as(next_prob) # beam_hold, 98

        next_prob=next_prob+log_prob_tensor  # beam_hold,54
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if i<=config.min_len:
            next_prob[:,config.eos_idx]=torch.full((next_prob.size(0),),float('-inf'))
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        next_prob=next_prob.view(-1) # beam_hold*98
        log_prob, indexes = torch.topk(next_prob, beam_hold) # [beam_hold]
        new_res=[]
        new_log_prob=[]
        for i in range(beam_hold):
            x=indexes[i].item()
            num=x//config.src_vocab_size
            last=x%config.src_vocab_size
            y=ans[num]+[last] # 4
            if last==eos:
                if len(y)>config.min_len:
                    if check(y): 
                        decode_ans.append((log_prob[i].item(),y))
                continue
            new_res.append(y)
            new_log_prob.append(log_prob[i].item())
    decode_ans.sort(reverse=True)

    decode_ans=decode_ans[:topk] # (prob,pw)
    return decode_ans

