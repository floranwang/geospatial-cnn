# geospatial-cnn
classify aerial/satellite imagery with a convolutional neural network via TFHub transfer learning

## Summary
The goal of this project is to reduce the number of out of scope establishments by identifying establishments that do not have shipment. This approach uses aerial imagery gathered internally and implements a convolutional neural network (CNN) to distinguish between establishments labeled as headquarters or as in-scope (having shipping activity). The best performing model, MobileNet v2 via a feature extractor, includes augmented training data and has a validation accuracy of 86%. 

## Data Collection
### Snapping adjusted coordinates from building footprints
First, we geocoded all of the establishments. Using these address-based coordinates, we found adjusted coordinates to reflect the center of a building. We did this by finding the building closest to the address-based coordinate. Then, we took the centroid of the building polygon. The building footprints data are licensed by Microsoft and freely available. 
With the new set of coordinates, we are able to more accurately query the aerial imagery to obtain an image with the building of the establishment at the center of the image (as opposed to the address-based coordinate refer to the edge of a building, in which the query could cut off the building in extracting the image).

### Data Augmentation
There are various data augmentation techniques for CNNs, particularly for unbalanced data. To balance the data (to improve test accuracy, precision, and recall), the images of HQ establishments were duplicated and rotated 90 and 180 degrees.

## Building CNN Model
This project used the Keras package from Tensorflow. We used a variety of different approaches as CNNs are relatively new and developing (see “Other Attempted Approaches” section). The model that we ultimately decided on was using a feature extractor based on the pretrained model, MobileNet v2.
The model uses the “adam” optimizer, which controls the learning rate, the “categorical_crossentropy” loss function, and “accuracy” as the metric. We trained the model on both the augmented and not augmented data.
