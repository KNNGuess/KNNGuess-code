import os
import logging
from collections import defaultdict
import math
from word2keypress import Keyboard
import config
kb=Keyboard()
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def similar(pw1,pw2):
    p1="|"+pw1+'\t'
    p2="|"+pw2+'\t'
    p1_dict=defaultdict(int)
    p2_dict=defaultdict(int)
    for i in range(len(p1)-1):
        p1_dict[p1[i]+p1[i+1]]+=1
    for i in range(len(p2)-1):
        p2_dict[p2[i]+p2[i+1]]+=1
    
    num=0
    den1=den2=0
    for pp in p1_dict:
        num+=p1_dict[pp]*p2_dict[pp]
        den1+=p1_dict[pp]*p1_dict[pp]
    for pp in p2_dict:       
        den2+=p2_dict[pp]*p2_dict[pp]
    
    return num/(math.sqrt(den1)*math.sqrt(den2))


# input: password1
# output: [('L','password'),('D','1')]
def get_segment(pw):
    res=""
    for chr in pw:
        if chr.isalpha():
            res+="L"
        elif chr.isdigit():
            res+="D"
        else:
            res+="S"
    cnt=0
    ans=[]
    for i in range(len(res)):
        cnt+=1
        if i+1==len(res) or res[i]!=res[i+1]:
            ans.append((res[i],pw[i-cnt+1:i+1]))
            cnt=0
    return ans

def check(pw):
    flag1=0
    flag2=0
    for c in pw:
        if c.isalpha():flag1=1
        if c.isdigit():flag2=1
    return flag1==1 and flag2==1

def get_mod_rate():
    f=open(config.train_data_path,"r",encoding='utf-8')
    if config.dataset_have_email:
        a,b=1,2
    else:
        a,b=0,1
    cnt,cnt_sim=0,0
    for line in f:
        cnt+=1
        lis=line.strip('\n').split('\t')
        pw1=kb.word_to_keyseq(lis[a])
        pw2=kb.word_to_keyseq(lis[b])
        if similar(pw1,pw2)>=0.3:
            cnt_sim+=1
    return cnt_sim/cnt
def get_popular_pws():
    popular_pws=[]
    cnt=0
    mod_rate=get_mod_rate()
    #print(mod_rate)
    cc7=math.log(mod_rate)
    if config.language=='cn':path="./tgaux_cn_toppsw.txt"
    else:path="./tgaux_en_toppsw.txt"
    f=open(path,"r",encoding='utf-8')
    for line in f:
        if cnt==1000:break
        lis=line.strip('\n').split('\t')
        if len(lis[0])>=config.min_len and check(lis[0]):
            popular_pws.append((lis[0],math.log(float(lis[1]))-cc7))
            cnt+=1
    return popular_pws[:350]

if __name__=='__main__':
    # p=get_popular_pws()
    # p=set([x[0] for x in p])
    # # top=set()
    # # f=open("tgaux_cn_toppsw.txt","r",encoding='utf-8')
    # # for idx,line in enumerate(f):
    # #     if idx==1000:break
    # #     lis=line.strip('\n').split('\t')
    # #     top.add(lis[0])
    # cnt=0
    # all_cnt=0
    # f=open("work_dataset/exp2/126_csdn_test.txt","r",encoding='utf-8')
    # for line in f:
    #     all_cnt+=1
    #     lis=line.strip('\n').split('\t')
    #     if lis[2] in p:
    #         cnt+=1
    # print(cnt/all_cnt)

    print("ADC".lower())