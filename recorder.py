import os
import requests
import threading
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import wave

import utils
import train

HAN2PIN = None
_txtPath = os.path.join(utils.cur_path,'data','han2pin.txt')
if os.path.exists(_txtPath):
    with open(_txtPath, 'r', encoding='utf8') as f:
        data = f.readlines()
        HAN2PIN = {l[0]:l[1] for l in [line.replace('\n','').split(':') for line in data]}
else:
    HAN2PIN={}
    with open(_txtPath, 'w', encoding='utf8') as f:
        tmp_pny,tmp_han = utils.make_all_file()
        for i,line in enumerate(tmp_han):
            #line = ''.join(line.split(' '))
            for j,han in enumerate(line):
                if HAN2PIN.get(han) is None:
                    HAN2PIN[han] = tmp_pny[i][j]
        for k,v in HAN2PIN.items():
            f.write('{}:{}\n'.format(k,v))


class FileRecord():
    def __init__(self,CHUNK=400,RATE=16000):
        self.filename = None
        self.allowRecording = False
        self.CHUNK = CHUNK
        self.RATE = RATE
        self.ani = SubplotAnimation(fun_use=True)
        self.wav_list=[]
        self.label_list=[]
        self.intUI()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.mainloop()


    def intUI(self):
        self.root = tkinter.Tk()
        self.root.title('wav音频录制')
        x = (self.root.winfo_screenwidth()-200)//2
        y = (self.root.winfo_screenheight()-140)//2
        self.root.geometry('430x200+{}+{}'.format(x,y))
        self.root.resizable(False,False)
        self.btStart = tkinter.Button(self.root,text='开始录音',command=self.start)
        self.btStart.place(x=50,y=20,width=100,height=40)
        self.btStop = tkinter.Button(self.root,text='停止录音',command=self.stop)
        self.btStop.place(x=50,y=80,width=100,height=40)
        self.btShowWav = tkinter.Button(self.root,text='Real-Time Wav',command=self.ShowWav)
        self.btShowWav.place(x=180,y=20,width=200,height=40)
        self.btTrain = tkinter.Button(self.root,text='开始训练',command=self.real_time_train)
        self.btTrain.place(x=180,y=80,width=100,height=40)
        self.btSumTrain = tkinter.Button(self.root,text='总体训练',command=self.sum_train)
        self.btSumTrain.place(x=280,y=80,width=100,height=40)
        self.label = tkinter.Text()
        self.label.place(x=50,y=140,width=330,height=40)
    

    def real_time_train(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.RATE,
                            input=True, frames_per_buffer=self.CHUNK)
        data = []
        while True:
            y = np.frombuffer(stream.read(self.CHUNK), dtype=np.int16)
            data.append(y)
            if self.ani._valid(np.array(data[-80::1]).flatten()):
                if len(data)>10:
                    txt = self.label.get('0.0', 'end').replace('\n','')
                    if txt.isalpha():
                        label = [HAN2PIN[h] for h in txt]
                        print('对应拼音为：{}'.format(label))
                    else:
                        label = txt.split(',')
                    wav = np.array(data).flatten()
                    pin = self.ani.yysb.predict(wav,only_pinyin=True)
                    print('【训练之前】预测拼音：{}'.format(pin))
                    train.train_am(wav,label)
                    self.wav_list.append(wav)
                    self.label_list.append(label)
                    #pin = self.ani.yysb.predict(wav,only_pinyin=True)
                    #print('【训练之后】预测拼音：{}'.format(pin))#目前有个问题训练前后结果一样，todo
                    break
                data=data[-9:]
        stream.stop_stream()
        stream.close()
        p.terminate()
    

    def sum_train(self):
        train.train_am(self.wav_list,self.label_list)


    def start(self):
        self.filename = tkinter.filedialog.asksaveasfilename(filetypes=[('Sound File','*.wav')])
        if not self.filename:
            return
        if not self.filename.endswith('.wav'):
            self.filename = self.filename+'.wav'
        self.allowRecording = True
        self.root.title('正在录音...')
        threading.Thread(target=self.record).start()


    def stop(self):
        self.allowRecording = False
        self.root.title('wav音频录制')


    def ShowWav(self):
        self.ani = SubplotAnimation(serviceAddress='http://172.16.100.213:20000/')
        plt.show()

    
    def close(self):
        if self.allowRecording:
            tkinter.messagebox.showerror('正在录音','请先停止录音')
            return
        self.root.destroy()


    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format = pyaudio.paInt16,channels=1,rate = self.RATE,
                        input = True,frames_per_buffer=self.CHUNK)
        wf = wave.open(self.filename,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.RATE)
        while self.allowRecording:#从录音设备读取数据，直接写入wav文件
            data = stream.read(self.CHUNK)
            wf.writeframes(data)
        wf.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.filename = None



class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, path = None,serviceAddress='http://127.0.0.1:20000/',fun_use=False):
        self.httpService = serviceAddress
        #音频波形动态显示，实时显示波形，实时进行离散傅里叶变换分析频域
        if path is not None and os.path.isfile(path):
            self.stream = wave.open(path)
            self.rate = self.stream.getparams()[2]
            self.chunk = int(self.rate/1000*25)
            self.read = self.stream.readframes
        else:
            self.rate = 16000
            self.chunk = 400#25*16000/1000针对语音识别25ms为一块这里相同设置
            p = pyaudio.PyAudio()
            self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.rate,
                            input=True, frames_per_buffer=self.chunk)
            self.read = self.stream.read
        self.yysb = utils.SpeechRecognition(test_flag=False)
        '''
        self.data说明：
        按时调用时：
        用来记录一整段话的数据，当听到明显声音开始填充，每次都把整个的内容送给语音识别，以期达到效果为：
        你
        你好
        你好啊
        当一个指定时间内没有明显声音时则清空
        自动判断启停时：
        从判断开始的数据开始记录，直到判断停止说话准备清空数据前调用一次API，效果：
        你好啊
        '''
        #self.data=np.ndarray(shape=(0), dtype=np.int16)
        self.data = []
        self.resHan=[]#语音识别结果，类型待定

        fig = plt.figure(num='Real-time wave')
        ax1 = fig.add_subplot(2, 1, 1)#两行一列，第一子图
        ax2 = fig.add_subplot(2, 1, 2)#两行一列，第二子图

        self.t = np.linspace(0, self.chunk - 1, self.chunk)
        #ax1.set_xlabel('t')
        #ax1.set_ylabel('x')
        self.line1, = ax1.plot([], [], lw=2)
        ax1.set_xlim(0, self.chunk)
        ax1.set_ylim(-6000, 6000)

        self.line2, = ax2.plot([], [], lw=2)
        ax2.set_xlim(0, self.chunk)
        ax2.set_ylim(-10, 50)

        interval = int(1000*self.chunk/self.rate)#更新间隔/ms
        if not fun_use:
            animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)


    def _valid(self,check_wav):
        '''
        判断是否开始、停止记录声音的方法，返回布尔结果
        if处可能需要根据情况设计更好的判断条件
        当返回为True时，开始、停止记录声音，False则记录声音
        '''
        #check = np.array([abs(x) for x in check_wav]).sum()/48000
        if check_wav.max()<900 and check_wav.min()>-900:#未听到声音
        #if check > 20:
            return True
        else:
            return False


    def _draw_frame(self, framedata):
        x = np.linspace(0, self.chunk - 1, self.chunk)
        y = np.frombuffer(self.read(self.chunk), dtype=np.int16)
        special_flag = False#特殊判断标记，当最后一段音频不足时赋值为真，主要就是针对读取固定长度音频的情况
        if len(y) == 0:
            return
        if len(y)<self.chunk:
            y = np.pad(y,(0,self.chunk-len(y)),'constant')#数据维度需要和坐标维度一致
            special_flag = True
        self.data.append(y)
        #默认最短3秒为每段话的间隔 3*1000/25=120：只要说话内容间隔3秒以上即清除之前的
        if special_flag or self._valid(np.array(self.data[-80::1]).flatten()):
            #修改语音识别调用方式：这种是在开始记录有效声音后直到准备清理数据时最后用完整数据调用一次
            if len(self.data)>10:
                wav = np.array(self.data).flatten()
                if True:#本地方式
                    pin,han = self.yysb.predict(wav)
                    print('识别拼音：{}'.format(pin))
                else:#发送到服务器的方式
                    try:
                        han = requests.post(self.httpService, {'token':'SR', 'data':wav,'pre_type':'H'})
                        han.encoding='utf-8'
                        han = han.text
                    except BaseException as e:
                        han = str(e)
                self.resHan.append(han)#记录用
                print('识别汉字：{}'.format(han))#todo:或者给需要的地方

            self.data=self.data[-9:]
            self.resHan.clear()
        # 波形图(上面的)
        self.line1.set_data(x, y)
        # 时频图（下面的）
        freqs = np.linspace(0, self.chunk, self.chunk / 2)
        _,_,xfp = utils.get_wav_Feature(wavsignal=y)
        self.line2.set_data(freqs, xfp)
        self._drawn_artists = [self.line1, self.line2]


    def new_frame_seq(self):
        return iter(range(self.t.size))


    def _init_draw(self):
        lines = [self.line1, self.line2]
        for l in lines:
            l.set_data([], [])



if __name__ == "__main__":
    ani = SubplotAnimation()
    plt.show()
    #rec = FileRecord()