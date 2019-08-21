#coding=utf-8
import os
import utils

def testModel():
    import difflib
    import tensorflow as tf
    import numpy as np
    yysb = utils.SpeechRecognition()

    # 1. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
    data_args = utils.data_hparams()
    data_args.data_type = 'test'
    data_args.shuffle = True
    data_args.batch_size = 1
    test = utils.get_data(data_args)

    # 2. 进行测试-------------------------------------------
    word_num = 0
    word_error_num = 0
    for i in range(100):
        print('\n 第 ', i, ' 个例子')
        label = test.han_lst[i]
        pinyin,hanzi = yysb.predict(os.path.join(test.data_path,test.wav_lst[i]),test.pny_lst[i],label,come_from_file=True)
        #hanzi = yysb.testPinyin(' '.join(test.pny_lst[i]))#单独测试语言模型用
        word_error_num += min(len(label), utils.GetEditDistance(label, hanzi))
        word_num += len(label)
    print('词错误率：', word_error_num / word_num)


def testClient():
    import requests
    import scipy
    data_args = utils.data_hparams()
    test = utils.get_data(data_args)

    _,wav = scipy.io.wavfile.read(os.path.join('/media/yangjinming/DATA/Dataset',test.wav_lst[0]))
    datas={'token':'SR', 'data':wav,'pre_type':'H'}
    r = requests.post('http://127.0.0.1:20000/', datas)
    r.encoding='utf-8'
    print(r.text)


if __name__ == "__main__":
    a=input('1.测试模型 2.测试服务端 3.都测试    请选择：')
    if a == '1' or a=='3':
        testModel()
    if a == '2' or a=='3':
        testClient()