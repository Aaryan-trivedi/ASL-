from keras import backend as K
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r'C:\Users\Rajesh Vishwakarma\Desktop\project\model\ResNet50_ASL6.h5')

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

# Testing
test = np.random.random(1, 480, 640, 3)[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)