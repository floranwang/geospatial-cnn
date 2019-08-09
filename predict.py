## Predict classifications of establishments from trained model

# imports and functions
import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
from sklearn.preprocessing import binarize
from tensorflow.keras import layers

import os

os.chdir("working directory")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    taken from https://github.com/scikit-learn/scikit-learn/blob/master/examples/model_selection/plot_confusion_matrix.py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def get_true_labels(label_batch):
    """
    This function returns a np array of labels from the tensorflow label_batch format.
    """
    true_labels = np.zeros((len(label_batch),), dtype = int)
    for i in range(len(label_batch)):
        x = label_batch[i]
        if int(x[1]) == 1:
            true_labels[i] = 1
    return true_labels



def getDF(all_image_paths):
    """
    This function returns a Pandas dataframe of the file paths of the images and the corresponding labels.
    """
    for path in all_image_paths:
        if not path.endswith(".png"):
            print(path)
            all_image_paths.remove(path)
    
    # labels of images
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_names

    # assign index for each label
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    label_to_index

    all_labels = [pathlib.Path(path).parent.name
                        for path in all_image_paths]

    df = pd.DataFrame({
        "filename" : all_image_paths,
        "class" : all_labels
    })
    
    return df



def getPredictions(image_data, threshold, allShipping = False):
    """
    This function returns np arrays of true labels, predicted labels, and predicted probabilities.
    image_data: generated from Keras image generator, in batch format
    threshold: the probability at which a classification should be considered shipping (1)
    allShipping: whether all image_data has a true shipping classification (eg. for the PHMSA data that is assumed to all have shipping activity)
    """
    all_true = np.zeros(0)
    all_pred = np.zeros(0)
    pred_prob = np.zeros(0)

    for i in range(len(image_data)):
        image_batch,label_batch = image_data[i]

        if (not allShipping):
            all_true = np.append(all_true, get_true_labels(label_batch))

        y_pred_prob = model.predict_proba(image_batch)[:,1]
        y_pred_class = binarize([y_pred_prob], threshold)[0]

        all_pred = np.append(all_pred, y_pred_class)
        pred_prob = np.append(pred_prob, y_pred_prob)
    
    if (allShipping):
        all_true = np.repeat(1, len(all_pred))
        
    return all_true, all_pred, pred_prob


## Create model and load trained weights
# Returns a short sequential model
def create_model():
    model = tf.keras.models.Sequential([
        features_extractor_layer,
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.train.AdamOptimizer(), 
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3"

# Create the module, and check the expected image size:
def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

# Wrap the module in a keras layer
features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])


# Create a basic model instance
model = create_model()

# initialize TFHub modules
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)

model.load_weights("saved_models/weights.h5")
model.summary()


## Load data for feature extractor model
data_root = pathlib.Path("hq_ship_images")
print(data_root)

all_image_paths = list(data_root.glob('*/*.png'))
all_image_paths = [str(path) for path in all_image_paths]
print(len(all_image_paths))

df = getDF(all_image_paths)


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_dataframe(df, target_size=(224,224), shuffle = False)


class_names = ["HQ","Shipping"]

# predict subset of images for image visualization
image_batch, label_batch = image_data[50]
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = [class_names[i] for i in predicted_id]

label_id = get_true_labels(label_batch)


# plot subset of images and the corresponding model predictions
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=0.2)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")



all_true, all_pred, pred_prob = getPredictions(image_data, 0.5, allShipping = False)

plot_confusion_matrix(all_true, all_pred, classes=class_names,
                      title='Confusion matrix ')



# outputs model evaluation metrics
from sklearn import metrics

accuracy = metrics.accuracy_score(all_true, all_pred) # (TP + TN) / (TP + TN + FP + FN)
precision = metrics.precision_score(all_true, all_pred) # TP / (TP + FP)
recall = metrics.recall_score(all_true, all_pred) # TP / (TP + FN)
f1 = metrics.f1_score(all_true, all_pred) # 2 * Precision * Recall / (Precision + Recall)

print(accuracy)
print(precision)
print(recall)
print(f1)


# ### Get predicted probability threshold accuracies
# evaulates accuracies based on different thresholds 
accs = []
steps = 50
thresh = 0
thresholds = []

for i in range(steps-1):
    thresh += 0.5/steps
    thresholds.append(thresh)
    indicies = [i for i in range(len(pred_prob)) if pred_prob[i] > thresh and pred_prob[i] < 1-thresh]
    
    sub_pred = pred_prob[indicies]
    sub_labels = all_true[indicies]
    
    sub_pred_class = binarize([sub_pred], 0.5)[0]
    accs.append(metrics.accuracy_score(sub_labels, sub_pred_class))


# plot accuracies
import matplotlib.pyplot as plt
accs.reverse()
plt.plot(thresholds, accs)
plt.show()


# plot distribution of predicted probabilities
plt.hist(pred_prob, bins=8, linewidth=1.2)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of shipping')
plt.ylabel('Frequency')


# ## Get csv of classifications
df['true_label'] = [int(num) for num in all_true]
df['pred_label'] = [int(num) for num in all_pred]
df['pred_prob'] = [int(num) for num in all_pred]
df['short_filename'] = [file.rsplit('\\', 1)[-1] for file in df['filename']] # grabs string after the last '\'
df.head()


# #### HQ + Shipping data
# load original data
hq = pd.read_csv("hq_augmented.csv")
hq.head()
shipping = pd.read_csv("shipping.csv")
shipping['filename'] = [str(shipping.loc[i, 'index']) + '.png' for i in range(len(shipping))]
shipping.head()


# Separate HQ and shipping classes
df_hq = df[df['class'] == 'hq']
df_shipping = df[df['class'] == 'shipping']


# Merge original data
df_hq_merged = df_hq.merge(hq, how = 'left', left_on = 'short_filename', right_on = 'filename')
df_hq_merged.head()


df_shipping_merged = df_shipping.merge(shipping, how = 'left', left_on = 'short_filename', right_on = 'filename')
df_shipping_merged.head()


output = df_hq_merged.append(df_shipping_merged)
output.head()
output.to_csv("results/hqShip_results.csv")


