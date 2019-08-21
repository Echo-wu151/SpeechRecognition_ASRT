import threading
import socket
import json
import numpy as np
import utils
import requests
import base64
import hashlib
import struct

'''
Python处理前端javascript发来的数据解码和编码参考以下内容，针对其中BUG有修改
https://blog.csdn.net/ice110956/article/details/34118203
'''
#支持的API类型，如果token不在list中则认为无效
API_Surport_List = ['SR']
ues_tf_serving = False
tf_serving_url = 'http://localhost:8501/v1/models/{}:predict'
if not ues_tf_serving:
	yysb = utils.SpeechRecognition(test_flag=False)


def SR_recognize(wavs,pre_type):
    hanzi =''
    am_url = tf_serving_url.format('am')
    lm_url = tf_serving_url.format('lm')
    if ues_tf_serving:
        x,_,_ = utils.get_wav_Feature(wavsignal=wavs)
        try:
            receipt = requests.post(am_url,data='{"instances":%s}' % x.tolist()).json()['predictions'][0]
            receipt = np.array([receipt],dtype=np.float32)
        except:
            return _,'声学模型调用异常'
        _, pinyin = utils.decode_ctc(receipt, utils.pny_vocab)
        pinyin = [[utils.pny_vocab.index(p) for p in ' '.join(pinyin).strip('\n').split(' ')]]
        if pre_type == 'H':
            #curl -d '{"instances": [[420,58]]}' -X POST http://localhost:8501/v1/models/lm:predict
            try:
                hanzi = requests.post(lm_url,data='{"instances": %s}' % pinyin).json()['predictions'][0]
            except:
                return _,'语言模型调用异常'
            hanzi = ''.join(utils.han_vocab[idx] for idx in hanzi)
    else:
        if pre_type == 'H':
            pinyin,hanzi = yysb.predict(wavs)
        else:
            pinyin = yysb.predict(wavs,only_pinyin = True)
    return pinyin,hanzi


def encode_data(data):
    data = data.encode('utf-8')
    token = b"\x81"
    length = len(data)
    if length < 126:
        token += struct.pack("B", length)
    elif length <= 0xFFFF:
        token += struct.pack("!BH", 126, length)
    else:
        token += struct.pack("!BQ", 127, length)
    #struct为Python中处理二进制数的模块，二进制流为C，或网络流的形式。
    data = token + data
    return data


def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    js_flag = False
    js_saver = b''#防止数据过长时单次接收不全
    while True:
        all_data = sock.recv(524288)
        if not all_data:
            break
        js_saver += all_data
        all_data = js_saver#all_data从js_saver处获取目前为止的全部数据
        try:
            datas = all_data.decode('utf-8')
        except:#处理前端js发来的websocket数据，还要进行解码处理
            js_flag = True
            code_len = all_data[1] & 127
            if code_len == 126:
                masks = all_data[4:8]
                all_data = all_data[8:]
            elif code_len == 127:
                masks = all_data[10:14]
                all_data = all_data[14:]
            else:
                masks = all_data[2:6]
                all_data = all_data[6:]
            datas = ""
            for i,d in enumerate(all_data):
                datas += chr(d ^ masks[i % 4])
        try:
            if datas.find('token') < 0 or datas.find('pre_type') < 0:
                continue
            datas = json.loads(datas)
            js_saver = b''
        except BaseException as e:
            print(e)
            continue
        token = datas['token']
        pre_type = datas['pre_type']
        receipt_data = list(datas['data'])
        
        if(token not in API_Surport_List):
            sock.send(b'token unsupported')
		
        if len(receipt_data)>0:
            if token == 'SR':
                wavs = np.array([int(w) for w in receipt_data])
                _,r = SR_recognize(wavs, pre_type)
            else:
                pass
        else:
            r = ''
            
        if js_flag:
            r = encode_data(r)
        else:
            r = r.encode('utf-8')
        sock.send(r)
    sock.close()
    print('Connection from %s:%s closed.\n' % addr)


if __name__ == "__main__":
    IPs = socket.gethostbyname_ex(socket.gethostname())[-1]
    # family=AF_INET - IPv4地址
    # family=AF_INET6 - IPv6地址
    # type=SOCK_STREAM - TCP套接字
    # type=SOCK_DGRAM - UDP套接字
    # type=SOCK_RAW - 原始套接字    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 监听端口:
    s.bind(('172.16.100.29', 9999))
    s.listen(255)# 参数255可以理解为连接队列的大小
    while True:
        # 接受一个新连接:
        # accept方法是一个阻塞方法如果没有客户端连接到服务器代码不会向下执行
        client, addr = s.accept()
        data = str(client.recv(1024))
        header_dict = {}
        header, _ = data.split(r'\r\n\r\n', 1)
        for line in header.split(r'\r\n')[1:]:
            key, val = line.split(': ', 1)
            header_dict[key] = val

        if 'Sec-WebSocket-Key' not in header_dict:
            print('This socket is not websocket, client close.')
            client.close()
            continue

        magic_key = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        sec_key = header_dict['Sec-WebSocket-Key'] + magic_key
        key = base64.b64encode(hashlib.sha1(bytes(sec_key, encoding='utf-8')).digest())
        key_str = str(key)[2:30]

        response = 'HTTP/1.1 101 Switching Protocols\r\n' \
                   'Connection: Upgrade\r\n' \
                   'Upgrade: websocket\r\n' \
                   'Sec-WebSocket-Accept: {0}\r\n' \
                   'WebSocket-Protocol: chat\r\n\r\n'.format(key_str)
        client.send(bytes(response, encoding='utf-8'))#针对普通js建立的websocket一定要回一个建立连接
        # 创建新线程来处理TCP连接:
        t = threading.Thread(target=tcplink, args=(client, addr))
        t.start()
