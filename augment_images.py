# ## Data Augmentation for CNN
# Duplicates and rotates images


import tensorflow as tf
import numpy as np
import os
import pandas as pd


def get_image_paths():
    """
    returns list of image paths
    """
    folder = 'hq_ship_images/hq'
    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files


img_paths = get_image_paths()


img_data = []
init = tf.initialize_all_variables()
sess = tf.Session()

with sess.as_default():
    tf.train.start_queue_runners()
    sess.run(init)

    for i in range(len(img_paths)):
        path = img_paths[i]

        image_string=tf.read_file(path)
        
        # loads image
        image=tf.image.decode_png(image_string,channels=3)
        
        # rotates image (angles parameter is in radians)
        #rot_tf_180 = tf.contrib.image.rotate(image, angles=3.1415)
        rot_tf_90 = tf.contrib.image.rotate(image, angles=1.5708)
        
        # save rotated image
        encoded = tf.image.encode_png(rot_tf_90)
        f = open("hq_ship_images/hq/ninty" + str(i) + ".png", "wb+")
        f.write(encoded.eval())
        f.close()



hq = pd.read_csv("hq.csv")
hq_90 = hq.copy()


# In[9]:


hq['filename'] = [str(i) + '.png' for i in range(len(hq))]
hq_90['filename'] = ['ninty' + str(i) + '.png' for i in range(len(hq_90))]


hq = hq.append(hq_90)
hq.head()

hq.to_csv("hq_augmented.csv")





