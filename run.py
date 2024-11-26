import os
import config
try:
    f=open(config.train_data_path,"r",encoding='utf-8')
except:
    print("error train data path")
    exit()
try:
    f=open(config.test_data_path,"r",encoding='utf-8')
except:
    print("error test data path")
    exit()
f.close()
cmd='python main.py'
os.system(cmd)
cmd='python gen_datastore.py'
os.system(cmd)
cmd='python test.py'
os.system(cmd)