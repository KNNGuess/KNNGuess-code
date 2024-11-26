import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from PW2SEQ import PW2SEQ
import config
from utils import similar,kb
DEVICE = config.device
pw2seq=PW2SEQ()
pw2seq_dict=pw2seq.pw2seq_dict


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=config.padding_idx):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg[:, :-1] 
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask





class MTDataset(Dataset):
    def __init__(self, data_path,mode='train'):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=config.sort_dataset,mode=mode)

        self.PAD=config.padding_idx
        self.BOS=config.bos_idx
        self.EOS=config.eos_idx

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False,mode='train'):
        dataset = open(data_path,"r",encoding='utf-8',errors='ignore')
        out_en_sent = []
        out_cn_sent = []
        if config.dataset_have_email==True:
            a,b=1,2
        else:
            a,b=0,1
        for line in dataset:
            lis=line.strip('\n').split('\t')
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            pw1=kb.word_to_keyseq(lis[a])
            pw2=kb.word_to_keyseq(lis[b])
            if mode=='train':
                if similar(pw1,pw2)>=0.3:
                    out_en_sent.append(pw1)
                    out_cn_sent.append(pw2)
            elif mode=='datastore':
                if similar(pw1,pw2)>=0.15:
                    out_en_sent.append(pw1)
                    out_cn_sent.append(pw2)
            else:
                out_en_sent.append(pw1)
                out_cn_sent.append(pw2)
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        input_text=self.out_en_sent[idx]
        target_text=self.out_cn_sent[idx]
        return [input_text,target_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):

        src_text=[x[0] for x in batch]
        tgt_text=[x[1] for x in batch]
        src_tokens = [[self.BOS] + pw2seq.transform(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + pw2seq.transform(sent) + [self.EOS] for sent in tgt_text]

        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                    batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD)



if __name__=='__main__':

    train_dataset = MTDataset(config.train_data_path)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=config.train_batch_size,
                                  collate_fn=train_dataset.collate_fn)
    for batch in train_dataloader:
        print(batch.src)
        print(batch.trg)
        break