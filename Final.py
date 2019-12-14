from PyQt5 import QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton,QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox,QLineEdit
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        self.setWindowTitle("Cifar10 Computer Vision")
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        self.col = QPushButton("Label Image", self)
        self.col.clicked.connect(self.update_ui)
        self.lab = QLabel("")
        self.img_lab = QLabel("The Image is of a: ")
        self.textbox = QLineEdit(self)
        self.header = QLabel("Computer vision")
        vbox.addWidget(self.header)
        vbox.addWidget(self.textbox)
        hbox.addWidget(self.img_lab)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        vbox.addWidget(self.col)
        self.labels ={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
        self.logits = None
        self.create_graph()

    @pyqtSlot()
    def update_ui(self):
        indx = self.get_label(self.textbox.text())
        res = self.labels[indx]
        self.textbox.setText("")
        self.img_lab.setText("was your image a/an " + res)

    def create_graph(self):
        height = 32
        width = 32
        channels = 3
        n_inputs = height * width * channels

        conv1_fmaps = 32
        conv1_ksize = 3
        conv1_stride = 1
        conv1_pad = "SAME"

        conv2_fmaps =64
        conv2_ksize = 3
        conv2_stride = 2
        conv2_pad = "SAME"


        conv3_fmaps = 128
        conv3_ksize = 4
        conv3_stride = 1
        conv3_pad = "SAME"
        pool3_fmaps = conv2_fmaps

        n_fc1 = 64
        n_outputs = 10
        
        tf.reset_default_graph()
        with tf.name_scope("inputs"):
            self.X = tf.placeholder(tf.float32, shape=[None, height,width,channels], name="X")
            self.X_reshaped = tf.reshape(self.X, shape=[-1, height, width, channels])
            self.y = tf.placeholder(tf.int32, shape=[None], name="y")

            self.conv1 = tf.layers.conv2d(self.X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                                     strides=conv1_stride, padding=conv1_pad,
                                     activation=tf.nn.selu, name="conv1")

            self.conv2 = tf.layers.conv2d(self.conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                                     strides=conv2_stride, padding=conv2_pad,
                                     activation=tf.nn.tanh, name="conv2")
            
        with tf.name_scope("pool3"):
            self.pool3 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
            self.pool3_flat = tf.reshape(self.pool3, shape=[-1, pool3_fmaps * 8  *8])
        
        with tf.name_scope("fc1"):
            self.fc1 = tf.layers.dense(self.pool3_flat, n_fc1, activation=tf.nn.selu, name="fc1")
            
        with tf.name_scope("output"):
            self.hidden_1 = tf.layers.dense(self.fc1,500,activation=tf.nn.tanh, name="hidden_1")
            self.hidden_2 = tf.layers.dense(self.hidden_1,400,activation=tf.nn.tanh, name="hidden_2")
            self.hidden_3 = tf.layers.dense(self.hidden_2,200,activation=tf.nn.tanh, name="hidden_3")
            self.hidden_4 = tf.layers.dense(self.hidden_3,200,activation=tf.nn.tanh, name="hidden_4")
            self.logits = tf.layers.dense(self.hidden_4, n_outputs, name="output")
            self.Y_proba = tf.nn.softmax(self.logits, name="Y_proba")
            
        with tf.name_scope("train"):
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            self.loss = tf.reduce_mean(self.xentropy)
            self.optimizer = tf.train.AdamOptimizer()
            self.training_op = self.optimizer.minimize(self.loss)
            
        with tf.name_scope("eval"):
            self.correct = tf.nn.in_top_k(self.logits, self.y, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32)) 
            
        with tf.name_scope("init_and_save"):
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.all_variables())
    def get_label(self,path):
        im = cv2.imread(path)
        im = im.reshape(1,32,32,3)
        with tf.Session() as sess:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess,"checkpoints/model")
            return tf.argmax(self.Y_proba.eval(feed_dict={self.X:im})[0]).eval()


        
        
        
        


app = QApplication(sys.argv)
window = MyApp()
window.show()
sys.exit(app.exec_())
