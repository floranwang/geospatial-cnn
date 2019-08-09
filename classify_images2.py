# # Build model to classify images with Mobile Net v2 Feature Extractor
# https://www.tensorflow.org/tutorials/images/hub_with_keras


from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import pathlib
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import binarize


data_root = pathlib.Path("file path")
print(data_root)


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, validation_split=0.25)

image_data_train = image_generator.flow_from_directory(str(data_root), target_size=(224,224), subset='training')
image_data_val = image_generator.flow_from_directory(str(data_root), target_size=(224,224), subset='validation')



for image_batch,label_batch in image_data_train:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break



feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3"


# Create the module, and check the expected image size:
def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))



# Wrap the module in a keras layer
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])


# Freeze the variables in the feature extractor layer, so that the training only modifies the new classifier layer.
features_extractor_layer.trainable = False



# Attach a classification head
model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data_train.num_classes, activation='softmax')
])
model.summary()



# initialize TFHub modules
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)


# configure training process
model.compile(
    optimizer=tf.train.AdamOptimizer(), 
    loss='categorical_crossentropy',
    metrics=['accuracy'])


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    
    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])



# train model
steps_per_epoch = image_data_train.samples//image_data_train.batch_size
batch_stats = CollectBatchStats()
model.fit_generator((item for item in image_data_train), epochs=7, 
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [batch_stats], validation_data = (item for item in image_data_val), 
          validation_steps =image_data_val.samples/image_data_val.batch_size)



model.save_weights('saved_models/weights.h5')


plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)


