####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#import wfdb
import os                                                                                                         
import gc
import scipy
import sklearn
#import seaborn as sns
#import shutil
import math
from sklearn import preprocessing
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from scipy.spatial import distance

####### Loading Dataset 

###### Loading Dataset - OSV
X_train = np.array(np.load('./Datasets/MIT_BIH/X_train_OSV_50-50_Test.npz',allow_pickle=True)['arr_0'],dtype=np.float16)
X_dev = np.array(np.load('./Datasets/MIT_BIH/X_dev_OSV_50-50_Test.npz',allow_pickle=True)['arr_0'],dtype=np.float16)
y_train = np.load('./Datasets/MIT_BIH/y_train_OSV_50-50_Test.npz',allow_pickle=True)['arr_0']
y_dev = np.load('./Datasets/MIT_BIH/y_dev_OSV_50-50_Test.npz',allow_pickle=True)['arr_0']

###### Converting Labels to Categorical Format
y_train_ohot = tf.keras.utils.to_categorical(y_train)
y_dev_ohot = tf.keras.utils.to_categorical(y_dev)


####### Model Making

###### Self-Calibrated Convolutions

###### Model Development : Self-Calibrated

##### Defining Self-Calibrated Block
rate_regularizer = 1e-5
class self_cal_Conv1D(tf.keras.layers.Layer):

    """ 
    This is inherited class from keras.layers and shall be instatition of self-calibrated convolutions
    """
    
    def __init__(self,num_filters,kernel_size,num_features):
    
        #### Defining Essentials
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_features = num_features # Number of Channels in Input

        #### Defining Layers
        self.conv2 = tf.keras.layers.Conv1D(self.num_features/2,self.kernel_size,padding='same',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32',activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(self.num_features/2,self.kernel_size,padding='same',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32',activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(self.num_filters/2,self.kernel_size,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32')
        self.conv1 = tf.keras.layers.Conv1D(self.num_filters/2,self.kernel_size,padding='same',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32')
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_features': self.num_features
        })
        return config
    
    
    def call(self,X):
       
        """
          INPUTS : 1) X - Input Tensor of shape (batch_size,sequence_length,num_features)
          OUTPUTS : 1) X - Output Tensor of shape (batch_size,sequence_length,num_features)
        """
        
        #### Dimension Extraction
        b_s = (X.shape)[0] 
        seq_len = (X.shape)[1]
        num_features = (X.shape)[2]
        
        #### Channel-Wise Division
        X_attention = X[:,:,0:int(self.num_features/2)]
        X_global = X[:,:,int(self.num_features/2):]
        
        #### Self Calibration Block

        ### Local Feature Detection

        ## Down-Sampling
        x1 = X_attention[:,0:int(seq_len/5),:]
        x2 = X_attention[:,int(seq_len/5):int(seq_len*(2/5)),:]
        x3 = X_attention[:,int(seq_len*(2/5)):int(seq_len*(3/5)),:]
        x4 = X_attention[:,int(seq_len*(3/5)):int(seq_len*(4/5)),:]
        x5 = X_attention[:,int(seq_len*(4/5)):seq_len,:]

        ## Convoluting Down Sampled Sequence 
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)
        x5 = self.conv2(x5)

        ## Up-Sampling
        X_local_upsampled = tf.keras.layers.concatenate([x1,x2,x3,x4,x5],axis=1)

        ## Local-CAM
        X_local = X_attention + X_local_upsampled

        ## Local Importance 
        X_2 = tf.keras.activations.sigmoid(X_local)

        ### Self-Calibration

        ## Global Convolution
        X_3 = self.conv3(X_attention)

        ## Attention Determination
        X_attention = tf.math.multiply(X_2,X_3)

        #### Self-Calibration Feature Extraction
        X_4 = self.conv4(X_attention)

        #### Normal Feature Extraction
        X_1 = self.conv1(X_global)

        #### Concatenating and Returning Output
        return (tf.keras.layers.concatenate([X_1,X_4],axis=2))

###### Transformer
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
    # add extra dimensions to add the padding
    # to the attention logits. 
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads':self.num_heads
        })
        
    def split_heads(self, x, batch_size):
        
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads':self.num_heads,
            'dff':self.dff,
            'maximum_position_encoding':self.maximum_position_encoding,
            'rate':self.rate  
        })
        
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def get_config(self):
        config = super(EncoderLayer, self).get_config().copy()
        config.update({
            'd_model': self.d_model,
            'num_heads':self.num_heads,
            'dff':self.dff,
            'rate':self.rate  
        })

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
        return out2
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                 pe_input, rate=0.1):
        super(Transformer, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.pe_input = pe_input
        self.rate = rate
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                                pe_input, rate)
        
    def get_config(self):
        config = super(Transformer,self).get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads':self.num_heads,
            'dff':self.dff,
            'pe_input':self.pe_input,
            'rate':self.rate  
        })
    
    def call(self, inp, training, enc_padding_mask):
        return self.encoder(inp, training, enc_padding_mask)

###### ArcFace Loss 
class ArcFace(tf.keras.layers.Layer):
    
    def __init__(self, n_classes, s, m,regularizer):
        super().__init__()
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True
                                )

    def call(self, inputs):
        x, y = inputs
        c = tf.keras.backend.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)


####### Model Training
###### Defining Architecture

#with tpu_strategy.scope():

##### SC_Module 

#### Defining Hyperparameters
num_layers = 2
d_model = 512
num_heads = 8
dff = 1024
max_seq_len = 1280 #X_train.shape[1]
pe_input = max_seq_len
rate = 0.5
num_features = 1
num_classes = 24

#### Defining Layers
Input_layer = tf.keras.layers.Input(shape=(max_seq_len,num_features))
self_conv1 = self_cal_Conv1D(128,3,128)
self_conv2 = self_cal_Conv1D(128,5,128) # Newly Added
self_conv3 = self_cal_Conv1D(256,3,128)
self_conv4 = self_cal_Conv1D(256,5,256) # Newly Added
self_conv5 = self_cal_Conv1D(512,3,256)
self_conv6 = self_cal_Conv1D(512,5,512) # Newly Added
self_conv7 = self_cal_Conv1D(1024,3,512)
self_conv8 = self_cal_Conv1D(1024,5,1024) # Newly Added
conv_initial = tf.keras.layers.Conv1D(32,3,padding='same',activation='relu')
conv_second = tf.keras.layers.Conv1D(64,3,padding='same',activation='relu')
conv_third = tf.keras.layers.Conv1D(128,3,padding='same',activation='relu')
transform_1 = tf.keras.layers.Conv1D(128,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_2 = tf.keras.layers.Conv1D(256,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_3 = tf.keras.layers.Conv1D(512,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transform_4 = tf.keras.layers.Conv1D(1024,3,padding='same',kernel_initializer='lecun_normal', activation='selu')
transformer = Transformer(num_layers,d_model,num_heads,dff,pe_input,rate)
gap_layer = tf.keras.layers.GlobalAveragePooling1D()
arc_logit_layer = ArcFace(24,30.0,1.57,tf.keras.regularizers.l2(1e-4))

#### Defining Architecture
### Input Layer
Inputs = Input_layer
Input_Labels = tf.keras.layers.Input(shape=(num_classes,))

### Initial Convolutional Layers
conv_initial = conv_initial(Inputs)
conv_second = conv_second(conv_initial)
conv_third = conv_third(conv_second)

### 1st Residual Block
transform_1 = transform_1(conv_third)
conv1 = self_conv1(conv_third)
conv2 = self_conv2(conv1)
conv2 = tf.keras.layers.Add()([conv2,transform_1])

### 2nd Residual Block
transform_2 = transform_2(conv2)
conv3 = self_conv3(conv2)
conv4 = self_conv4(conv3)
conv4 = tf.keras.layers.Add()([conv4,transform_2])

### 3rd Residual Block
transform_3 = transform_3(conv4)
conv5 = self_conv5(conv4)
conv6 = self_conv6(conv5)
conv6 = tf.keras.layers.Add()([conv6,transform_3])

### 4th Residual Block
#transform_4 = transform_4(conv6)
#conv7 = self_conv7(conv6)
#conv8 = self_conv8(conv7)
#conv8 = tf.keras.layers.Add()([conv8,transform_4])

### Transformer
## Wide-Head Attention Model
#tx_embedding = tf.keras.layers.Lambda(PE_Layer)(Inputs)
#tx_embedding = tf.keras.layers.Dropout(rate)(tx_embedding,training=True)
#mask_reshaped = tf.keras.layers.Reshape((max_seq_len,))(Inputs)
#encoder_op1 = encoder_block1(tx_embedding,mask_reshaped)
#encoder_op2 = encoder_block2(encoder_op1,mask_reshaped)

## Narrow-Head Attention Model
mask_reshaped = tf.keras.layers.Reshape((max_seq_len,))(Inputs)
embeddings = transformer(inp=conv6,enc_padding_mask=create_padding_mask(mask_reshaped))
#residual_embeddings = tf.keras.layers.Add()([conv6,embeddings])

### Output Layers
## Initial Layers
gap_op = gap_layer(embeddings)
dense1 = tf.keras.layers.Dense(256,activation='relu')(gap_op)
dropout1 = tf.keras.layers.Dropout(rate)(dense1)

## ArcFace Output Network
dense2 = tf.keras.layers.Dense(256,kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dropout1)
#dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense3 = arc_logit_layer(([dense2,Input_Labels]))

## Softmax Output Network
#dense2 = tf.keras.layers.Dense(256,activation='relu')(dropout1)
###dropout2 = tf.keras.layers.Dropout(rate)(dense2) # Not to be included
#dense3 = tf.keras.layers.Dense(48,activation='softmax')(dense2)

#### Compiling Architecture
### ArcFace Model Compilation
model = tf.keras.models.Model(inputs=[Inputs,Input_Labels],outputs=dense3)
### Softmax Model Compilation
#model = tf.keras.models.Model(inputs=Inputs,outputs=dense3)
model.load_weights('./Models/OSV_50-50_MITBIH_Test.h5')
model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()      
#tf.keras.utils.plot_model(model)
##### Model Training 

#### Model Checkpointing
filepath = './OSV_50-50_MIT-BIH_1.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)

#### Custom Learning Rate Schedule
#def build_lrfn(lr_start=1e-4, lr_max=1e-3, 
#               lr_min=1e-6, lr_rampup_epochs=5, 
#               lr_sustain_epochs=0, lr_exp_decay=.87):
#    lr_max = lr_max * tpu_strategy.num_replicas_in_sync

#    def lrfn(epoch):
#        if epoch < lr_rampup_epochs:
#            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
#        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
#            lr = lr_max
#        else:
#            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min#
#        return lr
#    
#    return lrfn

#lrfn = build_lrfn()
#lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
#callback_list = [checkpoint,  lr_callback]

#### Model Training
#### Model Training
### ArcFace Training
#history = model.fit((X_train,y_train_ohot),y_train_ohot,epochs=250,batch_size=128,
#                validation_data=((X_dev,y_dev_ohot),y_dev_ohot),validation_batch_size=128,
#                callbacks=checkpoint)

### Softmax Training 
#history = model.fit(X_train,y_train_ohot,epochs=250,batch_size=128,
#                validation_data=(X_dev,y_dev_ohot),validation_batch_size=128,
#                callbacks=checkpoint)

####### Embedding Generation

###### Testing Model
def normalisation_layer(x):
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

#X_dev_flipped = tf.image.flip_up_down(X_dev)
#x_train_flipped = tf.image.flip_up_down(X_train_final)

predictive_model = tf.keras.models.Model(inputs=model.input,outputs=model.layers[-3].output)
predictive_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

y_in = tf.keras.layers.Input((24,))

Input_Layer = tf.keras.layers.Input((1280,1))
op_1 = predictive_model([Input_Layer,y_in])

##Input_Layer_Flipped = tf.keras.layers.Input((224,224,3))
##op_2 = predictive_model([Input_Layer_Flipped,y_in]) 
##final_op = tf.keras.layers.Concatenate(axis=1)(op_1)

final_norm_op = tf.keras.layers.Lambda(normalisation_layer)(op_1)

testing_model = tf.keras.models.Model(inputs=[Input_Layer,y_in],outputs=final_norm_op)
testing_model.compile(tf.keras.optimizers.Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

###### Embedding Extraction
Test_Embeddings = testing_model.predict((X_dev,y_dev_ohot))
Train_Embeddings = testing_model.predict((X_train,y_train_ohot))

####### Matching File Generation
###### Matching File Generation
Matching_Matrix = [] # For Appending the Scores onto a List
match_text = open("./Scripts/Matching_Score_1/OSV_50-50_Test.txt",'w') # Matching Text File Creation   
num_probe_samples = 24 # Number of Subjects to be considered in Probe            
num_gallery_samples = 24 # Number of Subjects to be considered in Gallery                     

##### Probe Selection
for probe_class in range(num_probe_samples): # Looping over Probe Examples
    
    print('Currently Processing for Probe Class '+str(probe_class))

    #### Curating List of Examples belonging to Class 'probe_class'
    item_index = []
    for j in range(X_dev.shape[0]):
        if(y_dev[j] == probe_class):
            item_index.append(j)   

    for probe_id,probe_loc in enumerate(item_index):
        probe = Test_Embeddings[probe_loc] # Probe Selection 

        ##### Gallery Selection
        for gallery_class in range(num_gallery_samples): # Looping over Gallery Examples

            print('Currently Processing for Gallery Class '+str(gallery_class))

            #### Curating List of Examples belonging to Class 'probe_class'
            item_index = []
            for j in range(X_train.shape[0]):
                if(y_train[j] == gallery_class):
                    item_index.append(j)   

            for gallery_id,gallery_loc in enumerate(item_index):
                gallery = Train_Embeddings[gallery_loc] # Probe Selection 

                #### Metrics Computation 
                cos_distance = distance.cosine(probe,gallery) # Cosine Distance between Probe and Gallery
                
                if(probe_class == gallery_class): # Identity Flag
                    identity_flag = 1
                else:
                    identity_flag = 0
                current_matching_matrix = [probe_class+1,probe_id+1,gallery_class+1,gallery_id+1,identity_flag,cos_distance]
                
                ### For Array Appending       
                #Matching_Matrix.append(current_mathing_matrix)

                ### For Text File Appending
                for item_idx,item in enumerate(current_matching_matrix):
                    if(item_idx <= 4):
                        match_text.write(str(item)+'            ')
                    else:
                        match_text.write(str(item)+"\n")

