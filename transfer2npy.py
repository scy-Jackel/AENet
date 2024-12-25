import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

checkpoint_path='/home/cuiyang/bishe/CT-WGAN_AE_TF/WGAN_AE/code/AE/ckpt_ae/L067_3072/autoencoder.model-199999'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()


# only contain part of encoder.
autoencoder = {'AE/conv1_1/kernel':0,
               'AE/conv1_2/kernel':0,
               'AE/maxpool1':0,
               'AE/conv2_1/kernel':0,
               'AE/conv2_2/kernel':0,
               'AE/maxpool2':0,
               'AE/conv3_1/kernel':0,
               'AE/conv3_2/kernel':0,
               'AE/conv4_1/kernel':0,
               'AE/conv4_2/kernel':0,
               }

for key in var_to_shape_map:
    # print ("tensor_name",key)

    str_name = key
    # 因为模型使用Adam算法优化的，在生成的ckpt中，有Adam后缀的tensor
    if str_name.find('RMS') > -1:
        continue

    # print('tensor_name:' , str_name)
    if key in autoencoder.keys():
        autoencoder[key]=reader.get_tensor(key)
        print(key, 'saved. shape:', autoencoder[key].shape)


# save npy
np.save('autoencoder_3072.npy',autoencoder)
print('save npy over...')
#print(alexnet['conv1'][0].shape)
#print(alexnet['conv1'][1].shape)
