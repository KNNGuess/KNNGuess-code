import torch
import torch.nn as nn
from torch.autograd import Variable
from model import make_model
import logging
from tqdm import tqdm
from PW2SEQ import PW2SEQ
import config
import numpy as np
pw2seq=PW2SEQ()
from knn.datastore.datastore import Datastore
from knn.retriever.retriever import Retriever
from knn.combiner.combiner import Combiner
from lz_delete import beam_decode2
pw2seq_dict=pw2seq.pw2seq_dict
def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_data,  model, model_par, criterion, optimizer):
    best_bleu_score = 0.0
    #early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        train_loss = run_epoch(train_data, model_par,
                               MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        # model.eval()
        # dev_loss = run_epoch(dev_data, model_par,
        #                      MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        # bleu_score = evaluate(dev_data, model)
        # logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        
        torch.save(model.state_dict(), config.model_path)

        #logging.info("-------- Save Best Model! --------")


class LossCompute:

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize




def translate2(src, model, use_beam=True):
    #sp_chn = chinese_tokenizer_load()
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
    BOS = config.bos_idx  # 2
    EOS = config.eos_idx  # 3

    sent="adgwerrqfra"
    src_tokens = [[BOS] + pw2seq.transform(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    x=translate2(batch_input, model)