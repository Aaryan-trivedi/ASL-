from keras.layers import MaxPool2D, Conv2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from keras.applications.resnet import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import image
from keras.models import Sequential
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import os
import pandas as pd
import seaborn as sn

img_height, img_width = (96, 96)
batch_size = 32

train_data_dir = r"Random3\train"
test_data_dir = r"Random3\test"
val_data_dir = r"Random3\val"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.4)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

valid_generator = train_datagen.flow_from_directory(val_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')
test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                   target_size=(img_height, img_width),
                                                   batch_size=1,
                                                   class_mode='categorical',
                                                   subset='validation')

model = tf.keras.models.load_model(r'C:\Users\Rajesh Vishwakarma\Desktop\project\model\ResNet50_ASL6.h5')

# test_loss, test_acc = model.evaluate(test_generator, verbose=2)
# print('\nTest Accuracy:', test_acc)

filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob=[]
y_act=[]
test_generator.reset()
for _ in range(nb_samples):
    X_test,Y_test = test_generator.next()
    y_prob.append(model.predict(X_test))
    y_act.append(Y_test)

predicted_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
actual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act]

out_df = pd.DataFrame(np.vstack([predicted_class,actual_class]).T,columns=['predicted_class','actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'],out_df['predicted_class'],rownames=['Actual'],colnames=['Predicted'])

sn.heatmap(confusion_matrix,cmap='Blues', annot=True, fmt='d')
plt.show()
print('test accuracy : {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))