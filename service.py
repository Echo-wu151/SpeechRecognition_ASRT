import http.server
import urllib
import keras
import utils
import numpy as np
import socket
import requests
import json
'''
配置使用tensorflow serving的服务说明：

安装docker
安装docker镜像：
docker pull tensorflow/serving:latest-gpu
#docker pull tensorflow/serving #CPU版

sudo usermod -aG docker $USER 把当前用户添加到docker组
newgrp - docker 刷新docker组
sudo service docker restart 重启服务

#根据情况修改model.config中的配置

#调整source后的路径为对应的路径（model.config的上层目录）

sudo docker run --runtime=nvidia -p 8501:8501 \
 --mount type=bind,\
source=/media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/,\
target=/models \
 -t tensorflow/serving:latest-gpu \
 --model_config_file=/models/model.config
'''
#查看服务状态
#curl http://localhost:8501/v1/models/lm
#查看模型输入输出
#saved_model_cli show --dir /media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/logs_lm/190612/ --all
#关闭服务（把docker容器都杀了，请酌情修改）
#sudo docker ps | xargs sudo docker kill

#是否使用tensorflow serving服务，如果使用这个对外暴露的仅作为中转站
ues_tf_serving = True
#tensorflow serving的url地址,基本上只修改IP即可
tf_serving_url = 'http://localhost:8501/v1/models/{}:predict'
if not ues_tf_serving:
	yysb = utils.SpeechRecognition(test_flag=False)


class TestHTTPHandle(http.server.BaseHTTPRequestHandler):  
	def setup(self):
		self.request.settimeout(10)
		http.server.BaseHTTPRequestHandler.setup(self)
	

	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header("Access-Control-Allow-Headers", "Content-Type")
		self.end_headers()


	def do_GET(self):  
		buf = 'AI API'  
		self.protocal_version = 'HTTP/1.1'   
		
		self._set_response()
		buf = bytes(buf,encoding="utf-8")
		self.wfile.write(buf) 
	


	def do_OPTIONS(self):           
		self.send_response(200, "ok")
		self.send_header('Access-Control-Allow-Origin', '*')
		self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
		self.send_header("Access-Control-Allow-Headers", "Content-Type")


	def do_POST(self):  
		'''
		处理通过POST方式传递过来并接收的数据,通过 模型计算/调用tfserving 得到结果并返回
		'''             
		#获取post提交的数据
		datas = self.rfile.read(int(self.headers['content-length']))
		#datas = urllib.unquote(datas).decode("utf-8", 'ignore') 
		datas = datas.decode('utf-8')
		token = ''
		pre_type = 'H'
		receipt_data = []

		if datas.find('&') < 0:#处理x-www-form-unlencoded格式
			datas_json = json.loads(datas)
			token = datas_json['token']
			pre_type = datas_json['pre_type']
			receipt_data = list(datas_json['data'])
		else:
			datas_split = datas.split('&')
			for line in datas_split:
				[key, value]=line.split('=')
				if('data' == key and '' != value):
					receipt_data.append(value)
				elif('pre_type' == key):
					pre_type = value
				elif('token' == key ):
					token = value
				else:
					print(key, value)
		
		if len(receipt_data)>0:
			if token == 'SR':
				wavs = np.array([int(w) for w in receipt_data])
				_,r = self.SR_recognize(wavs, pre_type)
			else:
				buf = '403'
				buf = bytes(buf,encoding="utf-8")
				self.wfile.write(buf)  
				return
		else:
			r = ''

		self._set_response()
		r = bytes(r,encoding="utf-8")
		self.wfile.write(r)

	
	def SR_recognize(self, wavs,pre_type):
		hanzi =''
		if ues_tf_serving:
			am_url = tf_serving_url.format('am')
			lm_url = tf_serving_url.format('lm')
			x,_,_ = utils.get_wav_Feature(wavsignal=wavs)
			try:
				receipt = requests.post(am_url,data='{"instances":%s}' % x.tolist()).json()['predictions'][0]
				receipt = np.array([receipt],dtype=np.float32)
			except BaseException as e:
				return _,'声学模型调用异常'
			_, pinyin = utils.decode_ctc(receipt, utils.pny_vocab)
			pinyin = [[utils.pny_vocab.index(p) for p in ' '.join(pinyin).strip('\n').split(' ')]]
			if pre_type == 'H':
				#curl -d '{"instances": [[420,58]]}' -X POST http://localhost:8501/v1/models/lm:predict
				try:
					hanzi = requests.post(lm_url,data='{"instances": %s}' % pinyin).json()['predictions'][0]
				except BaseException as e:
					return _,'语言模型调用异常'
				hanzi = ''.join(utils.han_vocab[idx] for idx in hanzi)
		else:
			if pre_type == 'H':
				pinyin,hanzi = yysb.predict(wavs)
			else:
				pinyin = yysb.predict(wavs,only_pinyin = True)
		return pinyin,hanzi



class HTTPServerV6(http.server.HTTPServer):
	address_family = socket.AF_INET6



def start_server(ip, port):  
	if(':' in ip):
		http_server = HTTPServerV6((ip, port), TestHTTPHandle)
	else:
		http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)
	print('服务器已开启')
	try:
		http_server.serve_forever() #设置一直监听并接收请求  
	except KeyboardInterrupt:
		pass
	http_server.server_close()
	print('服务器关闭')



if __name__ == '__main__':
	start_server('', 5000) # For IPv4 Network Only
	#start_server('::', 20000) # For IPv6 Network