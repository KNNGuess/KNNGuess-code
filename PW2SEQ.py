# {'<UNK>': 0, '<PAD>': 1, '<SOS>': 2, '<EOS>': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, 
# '7': 11, '8': 12, '9': 13}
import config
import string
class PW2SEQ:
    UNK_TAG = '<UNK>'  # 特殊字符
    PAD_TAG = '<PAD>'  # 填充字符
    BOS_TAG = '<BOS>'  # 序列开始字符
    EOS_TAG = '<EOS>'  # 句子结束字符
    CHAR_SHIFT='\x03'
    CHAR_CAPSLOCK='\x04'
    PAD = config.padding_idx
    BOS = config.bos_idx
    EOS = config.eos_idx
    UNK = 6-PAD-BOS-EOS
    def __init__(self):
        self.pw2seq_dict={
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.BOS_TAG: self.BOS,
            self.EOS_TAG: self.EOS
        }
        #total_chars = string.digits + string.ascii_letters + '~`!@#$%^&*()_-+={[}]|\\:;\'"<,>.?/' #+ self.CHAR_SHIFT + self.CHAR_CAPSLOCK
        total_chars='`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./ '+ self.CHAR_SHIFT + self.CHAR_CAPSLOCK
         #print(len(total_chars))
        # pw2seq_dict:{'<UNK>': 0, '<PAD>': 1, '<SOS>': 2, '<EOS>': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13}
        for i in range(len(total_chars)):
            self.pw2seq_dict[total_chars[i]]=len(self.pw2seq_dict)
        self.sequenceToNum_dict=dict(zip(self.pw2seq_dict.values(),self.pw2seq_dict.keys()))

    # {'<UNK>': 0, '<PAD>': 1, '<SOS>': 2, '<EOS>': 3, '0': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10, '7': 11, '8': 12, '9': 13}
    # ['1','9','9','8'] -> [5, 13, 13, 12, 1]
    def transform(self,stringNum,max_len=config.max_len,add_eos=False): 
        #print(stringNum)
        # if add_eos:
        #     max_len-=1
        # if max_len is not None:
        #     if len(stringNum)>max_len:
        #         stringNum=stringNum[:max_len]
        #     else:
        #         stringNum+=[self.PAD_TAG]*(max_len-len(stringNum))
        # if add_eos:
        #     if stringNum[-1]==self.PAD_TAG:
        #         stringNum.insert(stringNum.index(self.PAD_TAG),self.EOS_TAG)
        #     else:
        #         stringNum+=[self.EOS_TAG] # 数字字符串中没有PAD,在最后添加EOS
        #print(stringNum)
        return [self.pw2seq_dict.get(charNum,self.UNK) for charNum in stringNum]
    
    # [5, 13, 13, 12, 1] -> ['1', '9', '9', '8', '<PAD>'] 
    def inverse_transform(self,sequence): 
        results=[]
        for index in sequence:
            result=self.sequenceToNum_dict.get(index,self.UNK_TAG)
            if result==self.BOS_TAG:
                continue
            if result!=self.EOS_TAG:
                results.append(result)
            else:
                break
        return "".join(results)
    
    def __len__(self):
        return len(self.pw2seq_dict)
    
if __name__=='__main__':
    pw2seq=PW2SEQ()
    print(pw2seq.inverse_transform([2,4,5]))