import os
import tensorflow as tf
import utils
from tqdm import tqdm
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

from datetime import datetime
modelVersion = str(datetime.now())[2:10].replace("-", "")


# 0.准备训练所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'train'
#data_args.thchs30 = True
#data_args.aishell = True
#data_args.prime = True
#data_args.stcmd = True
data_args.batch_size = 16#可以将不一次性训练am和lm，同样显存情况下lm的batch_size可以比am的大许多
train_data = utils.get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = utils.data_hparams()
data_args.data_type = 'dev'
#data_args.thchs30 = True
#data_args.aishell = True
#data_args.prime = True
#data_args.stcmd = True
data_args.batch_size = 16
dev_data = utils.get_data(data_args)


batch_num = len(train_data.wav_lst) // train_data.batch_size
epochs = 100#两个模型都加入了提前终止判断，可以大一些，反正又到不了

# 1.声学模型训练-----------------------------------
def train_am(x=None,y=None,fit_epoch=10):
    from model_speech.cnn_ctc import Am, am_hparams
    am_args = am_hparams()
    am_args.vocab_size = len(utils.pny_vocab)
    am_args.gpu_nums = 1
    am_args.lr = 0.0008
    am_args.is_training = True
    am = Am(am_args)

    if os.path.exists(os.path.join(utils.cur_path,'logs_am','model.h5')):
        print('加载声学模型...')
        am.ctc_model.load_weights(os.path.join(utils.cur_path,'logs_am','model.h5'))

    checkpoint = ModelCheckpoint(os.path.join(utils.cur_path,'checkpoint', "model_{epoch:02d}-{val_loss:.2f}.h5"),
        monitor='val_loss',save_best_only=True)
    eStop = EarlyStopping()#损失函数不再减小后patience轮停止训练
    #tensorboard --logdir=/media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/logs_am/tbLog/ --host=127.0.0.1
    #tensbrd = TensorBoard(log_dir=os.path.join(utils.cur_path,'logs_am','tbLog'))

    if x is not None:#利用实时声音训练调整模型，使定制化
        size=1
        if type(x) == np.ndarray:
            x,y = utils.real_time2data([x],[y])
        else:
            size = len(x)
            x,y = utils.real_time2data(x,y)
        am.ctc_model.fit(x=x,y=y,batch_size=size,epochs=fit_epoch)
    else:#利用训练数据
        batch = train_data.get_am_batch()#获取的是生成器
        dev_batch = dev_data.get_am_batch()
        validate_step = 200#取N个验证的平均结果
        history = am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[eStop,checkpoint],
            workers=1, use_multiprocessing=False,verbose=1,validation_data=dev_batch, validation_steps=validate_step)

    am.ctc_model.save_weights(os.path.join(utils.cur_path,'logs_am','model.h5'))
    #写入序列化的 PB 文件
    #with keras.backend.get_session() as sess:
    sess = keras.backend.get_session()
    constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
                            sess.graph_def,output_node_names=['the_inputs','dense_2/truediv'])
    with tf.gfile.GFile(os.path.join(utils.cur_path,'logs_am','amModel.pb'), mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    #保存TF serving用文件
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(utils.cur_path,'logs_am',modelVersion))
    model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={'input': am.inputs}, outputs={'output': am.outputs})
    builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature})
    builder.save()

    if x is None:
        sess.close()

    


# 2.语言模型训练-------------------------------------------
def train_lm(eStop = False):
    global epochs
    from model_language.transformer import Lm, lm_hparams
    import numpy as np
    lm_args = lm_hparams()
    lm_args.num_heads = 8
    lm_args.num_blocks = 6
    lm_args.input_vocab_size = len(utils.pny_vocab)
    lm_args.label_vocab_size = len(utils.han_vocab)
    lm_args.max_length = 100
    lm_args.hidden_units = 512
    lm_args.dropout_rate = 0.2
    lm_args.lr = 0.0003
    lm_args.is_training = True
    lm = Lm(lm_args)

    batch_num_list = [i for i in range(batch_num)]#为进度条显示每一个epoch中的进度用
    loss_list=[]#记录每一步平均损失的列表，实现提前终止训练功能：每次取出后N个数据的平均值和当前的平均损失值作比较
    with lm.graph.as_default():
        saver =tf.train.Saver()
    with tf.Session(graph=lm.graph) as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        if os.path.exists(os.path.join(utils.cur_path,'logs_lm','checkpoint')):
            print('加载语言模型中...')
            latest = tf.train.latest_checkpoint(os.path.join(utils.cur_path,'logs_lm'))
            add_num = int(latest.split('_')[-1])
            saver.restore(sess, latest)
        #tensorboard --logdir=/media/yangjinming/DATA/GitHub/AboutPython/AboutDL/语音识别/logs_lm/tensorboard --host=127.0.0.1
        #writer = tf.summary.FileWriter(os.path.join(utils.cur_path,'logs_lm','tensorboard'), tf.get_default_graph())
        for k in range(epochs):
            total_loss = 0
            batch = train_data.get_lm_batch()
            for i in tqdm(batch_num_list,ncols=90):
                input_batch, label_batch = next(batch)
                feed = {lm.x: input_batch, lm.y: label_batch}
                cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs=sess.run(merged, feed_dict=feed)
                    #writer.add_summary(rs, k * batch_num + i)
            avg_loss = total_loss/batch_num
            print('步数', k+1, ': 平均损失值 = ', avg_loss)

            loss_list.append(avg_loss)
            if eStop and len(loss_list)>1 and avg_loss>np.mean(loss_list[-5:])-0.0015:#平均每个epoch下降不到0.0005则终止
                #if input('模型性能已无法提升，是否提前结束训练？ yes/no:')=='yes':
                    epochs = k+1#为后面保存模型时记录名字用
                    break
        
        saver.save(sess, os.path.join(utils.cur_path,'logs_lm','model_%d' % (epochs + add_num)))
        #writer.close()
        # 写入序列化的 PB 文件
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['x','y','preds'])
        with tf.gfile.GFile(os.path.join(utils.cur_path,'logs_lm','lmModel.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())

        #tf serving 用保存文件，目前保存了三种模型，按需选择一种即可
        model_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"pinyin": tf.saved_model.utils.build_tensor_info(lm.x)},
            outputs={"hanzi": tf.saved_model.utils.build_tensor_info(lm.preds)},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(utils.cur_path,'logs_lm',modelVersion))
        builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
            {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature})
        builder.save()


if __name__ == "__main__":
    if input('训练声学模型？ yes/no:') == 'yes':
        train_am()
    if input('训练语言学模型？ yes/no:') == 'yes':
        train_lm()
