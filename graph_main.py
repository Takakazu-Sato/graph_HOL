from __future__ import print_function

import os
import sys
import traceback_utils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.opt as tf_opt
import datetime
from tensorflow.python.keras.layers import Input, Dense, Embedding, LSTM, SimpleRNN
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix


from data_utils import DataParser
from graph_conv import ConvNetwork
from graph_list import FormulaReader
from tf_utils import predict_loss_acc

class Network:

    def __init__(self, vocab_size, edge_arities,
                 step_signature = ((2,64), (2,128), (2,256)),
                 conj_signature = ((3,128), (3,192)),
                 ver2 = True):

        self.step_network = ConvNetwork(vocab_size, step_signature,
                                        edge_arities, ver2 = ver2)
        self.conj_network = ConvNetwork(vocab_size, conj_signature,
                                        edge_arities, ver2 = ver2)

    def construct(self, threads = 4):

        graph = tf.Graph()
        graph.seed = 42
        config = tf.ConfigProto(
            inter_op_parallelism_threads=threads,
            intra_op_parallelism_threads=threads,
            #device_count = {'GPU': 0},
        )
        self.session = tf.Session(graph = graph, config = config)
        with self.session.graph.as_default():

            #graph_conv.pyのConvNetworkインスタンス
            #ConvNetworkインスタンスのdataがstepとconjに代入される．
            with tf.name_scope("Step"):
                step = self.step_network() # [bs, dim]

            with tf.name_scope("Conjecture"):
                conj = self.conj_network() # [bs, dim]

            step_conj = tf.concat([step, conj], axis = 1)

            hidden = tf_layers.fully_connected(step_conj, num_outputs=256, activation_fn = tf.nn.relu)
            hidden = tf_layers.dropout(hidden,0.5)

            self.logits = tf_layers.fully_connected(hidden, num_outputs=2, activation_fn = None)

            self.labels = tf.placeholder(tf.int32, [None])

            self.predictions, self.loss, self.accuracy = predict_loss_acc(self.logits, self.labels)

            #self.training = tf_opt.AdaMaxOptimizer(0.001).minimize(self.loss)
            #self.training = tf_opt.NadamOptimizer(0.001).minimize(self.loss)
            self.training = tf.train.AdamOptimizer(0.001).minimize(self.loss)

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

        # Finalize graph and log it if requested
        self.session.graph.finalize()

    def feed(self, steps, conjectures, labels ):
    #def feed(self, steps, conjectures, labels = None):
        data = self.step_network.feed(steps)
        data.update(self.conj_network.feed(conjectures))
        if labels is not None:
            #同じキーの要素については値が更新され、存在していないキーの要素は新しい要素として追加される
            data.update({ self.labels: labels })
        return data

    def train(self, steps, conjectures, labels):
        data = self.feed(steps, conjectures, labels)
        print("Train")
        _, accuracy, loss = self.session.run(
            [self.training, self.accuracy, self.loss],
            data,
        )

        return accuracy, loss

    def evaluate(self, steps, conjectures, labels):
        return self.session.run(
            [self.accuracy, self.loss],
            self.feed(steps, conjectures, labels)
        )

    #def predict(self, steps, conjectures):
    def predict(self, steps, conjectures,labels):
        return self.session.run(
            self.predictions,
            self.feed(steps, conjectures, labels)
        )

    def spline_interp(self,in_x, in_y):
        out_x = np.linspace(np.min(in_x), np.max(in_x), np.size(in_x)*100)
        func_spline = interp1d(in_x,in_y,kind='cubic')
        out_y = func_spline(out_x)
        return out_x,out_y

encoder = FormulaReader(ver2 = True)
data_parser = DataParser("./e-hol-ml-dataset/", encoder = encoder,
                         ignore_deps = True, truncate_test = 1, truncate_train = 1) #0.05:0.01

#vocab_size=486,edge_arities=(3,3)
network = Network(
    encoder.vocab_size, encoder.edge_arities,
    #step_signature = ((2,32), (2,64), (2,128)),
    #conj_signature = ((2,64), (2,128)),
    ver2 = encoder.ver2,
)
network.construct()

# training

batch_size = 256
#index = (0,0)
acumulated = 0.5
epoc = 2000
list_train_acc = []
list_train_loss = []
list_epoc= []
list_vali_acc = []
list_vali_loss = []
list_count= []
#list_labels= []
#list_predict= []
sum_real= []
sum_predictions= []

for i in range(epoc):

    print("Prepare data")

    batch = data_parser.draw_batch(
        batch_size=batch_size,
        split='train',
        get_conjectures = True,
        use_preselection = False,
        #begin_index = index
    )

    numlabels = len(batch['labels'])

    print("batch['labels']:{}".format(batch['labels']))

    acc, loss = network.train(
        batch['steps'],
        batch['conjectures'],
        batch['labels'],
    )
    print("training_acc:{},  loss:{}".format(acc,loss))

    list_train_acc.insert(epoc,acc)
    list_train_loss.insert(epoc,loss)

    list_epoc.append(i+1)

    acumulated = acumulated*0.99 + acc*0.01

    if True or (i+1)%100 == 0: print("{}: {}".format(i+1, acumulated))

# testing

index = (0,0)
sum_accuracy = sum_loss = 0
processed_test_samples = 0
count = 1
batch_size = 256

while True:
    real_list = []
    predictions_list = []

    print("Prepare data for eval.")
    batch, index = data_parser.draw_batch(
    #batch, index = data_parser.draw_random_batch_of_steps(
        split='val',
        batch_size=batch_size,
        get_conjectures = True,
        use_preselection = False,
        begin_index = index,
    )

    numlabels = len(batch['labels'])
    if numlabels == 0: break

    print("batch['labels']:{}".format(batch['labels']))

    #print("Evaluate")
    #accuracy, loss = network.evaluate(
    predictions = network.predict(
        batch['steps'],
        batch['conjectures'],
        batch['labels'],
    )
    print("predictions:{}".format(predictions))
    #print("{}: {}".format(count,accuracy))

    #.append：リストも１つの要素として加えられる（結合はできない）
    #.extend：リストの結合
    #list_vali_acc.append(accuracy)
    #list_vali_loss.append(loss)
    #list_count.append(count)
    real_list = batch['labels'].tolist()
    predictions_list= predictions.tolist()

    #list_labels.append(batch['labels'])
    #list_predict.append(predictions)
    sum_real.extend(real_list)
    sum_predictions.extend(predictions_list)
    #count = count+1

print(confusion_matrix(sum_real,sum_predictions))
"""sum_accuracy += accuracy*numlabels
    sum_loss += loss*numlabels
    processed_test_samples += numlabels

    if numlabels < batch_size: break # Just a smaller batch left -> we are on the end of the testing dataset

print("Development accuracy: {}, avg. loss: {}".format(sum_accuracy/processed_test_samples, sum_loss/processed_test_samples))"""

#混同行列
# list_labels = [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]   # 実際の値 (0:-, 1:+)
# list_predict = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]   # 識別結果 (0:-, 1:+)
#confusion_matrix(list_labels, list_predict)
#confusion_matrix(a,b)
"""#x軸:エポック数,y軸:学習精度
x_e=list_epoc
y_t_a=list_train_acc
x1,y1=network.spline_interp(x_e,y_t_a)
plt.plot(x1,y1,label="train acc")
plt.ylim(0,1)
plt.title("Train Acc")
plt.legend()
plt.show()

y_t_l=list_train_loss
x2,y2=network.spline_interp(x_e,y_t_l)
plt.plot(x2,y2,label="train loss")
plt.ylim(0,4)
plt.title("Train Loss")
plt.legend()
plt.show()

#x軸:テスト数,y軸:精度
x_c=list_count
y_v_a=list_vali_acc
x3,y3=network.spline_interp(x_c,y_v_a)
plt.plot(x3,y3,label="vali acc")
plt.ylim(0,1)
plt.title("Validation Acc")
plt.xlabel("epoc")
plt.ylabel("acc")
plt.legend()
plt.show()
#loss
y_v_l=list_vali_loss
x4,y4=network.spline_interp(x_c,y_v_l)
plt.plot(x4,y4,label="vali loss")
plt.ylim(0,4)
plt.title("Validation Loss")
plt.legend()
plt.show()"""
