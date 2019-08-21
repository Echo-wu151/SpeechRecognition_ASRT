import utils
import json
import os
import numpy as np
'''
根据data中的txt生成另一个语音识别项目MASR所需的index文件
'''

# 生成全数据集的拼音和汉字列表
def make_all_file(datatype='train'):
    read_files = []
    read_files.append('thchs_{}.txt'.format(datatype))
    read_files.append('aishell_{}.txt'.format(datatype))
    read_files.append('prime_{}.txt'.format(datatype))
    read_files.append('stcmd_{}.txt'.format(datatype))
    path_lst,han_lst = [],[]
    for file in read_files:
        sub_path = os.path.join(utils.cur_path,'data')
        sub_file = os.path.join(sub_path,file)
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in data:
            wav, _, han = line.split('\t')
            path_lst.append(wav)
            han_lst.append(han.strip('\n'))
    return path_lst,han_lst


def write_file():
    for name in ['train','dev']:
        with open('data/{}.index'.format(name), 'w', encoding='utf8') as f:
            tmp_path,tmp_han = make_all_file(name)
            for i,p in enumerate(tmp_path):
                f.write('{},{}\n'.format(p,tmp_han[i]))

    label = np.load(os.path.join(utils.cur_path,'data','han_vocab_A.npy')).tolist()
    label[0]='_'
    with open('data/labels.json', 'w', encoding='utf-8') as fs:
        json.dump(label, fs)


if __name__ == "__main__":
    write_file()
    print('Finish!')