# Common
from datetime import date
import os
import sys
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

# Data 
import cv2 as cv
from glob import glob
from tqdm import tqdm
import tensorflow.data as tfd

# Data Visualization
import matplotlib.pyplot as plt

# Data Augmentation
from keras.models import Sequential
from keras.layers import Input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomContrast

# Model Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Model 
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D as GAP, Dense,MaxPooling2D, Dropout, BatchNormalization,GlobalAveragePooling2D, Flatten, Conv2D, DepthwiseConv2D, Activation,Reshape
from tensorflow.keras.applications import ResNet50, ResNet152, ResNet50V2, ResNet152V2, InceptionV3, MobileNet

#Own Model
def conv_relu(filters, kernel_size=(3,3), strides=(1, 1), padding='same'):
    layer = Sequential()
    layer.add(Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False))
    layer.add(BatchNormalization())
    layer.add(Activation('relu'))
    return layer
def depth_conv_relu(filters, kernel_size=(3,3), strides=(1, 1), padding='same'):
    layer = Sequential()
    layer.add(DepthwiseConv2D(filters = filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False))
    layer.add(BatchNormalization())
    layer.add(Activation('relu'))
    layer.add(Conv2D(filters = filters, kernel_size=(1,1), strides=strides, padding=padding, use_bias=False))
    layer.add(BatchNormalization())
    layer.add(Activation('relu'))
  

    return layer

def dense_relu(dropout = 0.001):
    layer = Sequential()
    layer.add(BatchNormalization())
    layer.add(Dense(256,activation='relu'))
    layer.add(Dropout(dropout))
    return layer

def aspiring_mobile_net(num_classes, dropout=0.001):
    model = Sequential(name ="TrulyMonstrousModel" )
    # Warstwy konwolucyjne
    model.add(conv_relu(filters=32, kernel_size=(3, 3), strides=(1, 1))) 
    model.add(conv_relu(filters=32, kernel_size=(3, 3), strides=(1, 1))) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv_relu(filters=64, kernel_size=(3, 3), strides=(1, 1))) 
    model.add(conv_relu(filters=64, kernel_size=(3, 3), strides=(1, 1))) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv_relu(filters=128, kernel_size=(3, 3), strides=(1, 1)))  
    model.add(conv_relu(filters=128, kernel_size=(3, 3), strides=(1, 1)))  
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv_relu(filters=256, kernel_size=(3, 3), strides=(1, 1))) 
    model.add(conv_relu(filters=256, kernel_size=(3, 3), strides=(1, 1)))      
    model.add(conv_relu(filters=256, kernel_size=(3, 3), strides=(1, 1)))     
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    model.add(conv_relu(filters=512, kernel_size=(3, 3), strides=(1, 1)))    
    # Dodatkowe warstwy konwolucyjne
    # Warstwa poolingowa
    model.add(GlobalAveragePooling2D())
    # Warstwa FC
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(dropout))

    return model
    


def collect_filepaths(root_path:str, class_names:list, shuffle:bool = True) -> list:
    '''
    
    It collects image paths from all classes and returns a shuffled list/list of all the file paths.
    
    Inputs :
        root_path   : The part to the main directory where the classes are stored.
        class_names : Name of the subdirectories is present in the main directory i.e The class names.
        shuffle     : True or false, depending on the need of shuffling the order of the file paths.
    
    Output :
        all_filepaths : It is a list of collections of all the paths.
        
    '''
    
    all_filepaths = []
    
    # Collect Filepaths for each class
    for name in tqdm(class_names, desc="Collecting "):
        class_path = root_path + f"{name}/"
        
        # Collect the filepath on this class
        filepaths = sorted(glob(class_path + "*.jpg"))
        
        # Store all paths
        for path in filepaths:
            all_filepaths.append(path)
        
    # Shuffle the file paths --> It is necessary
    if shuffle:
        np.random.shuffle(all_filepaths)
    
    # return the list
    return all_filepaths

def load_image(image_path:str, size:tuple = (256,256), normalize:bool = True):
    
    '''
    The function takes in the image path and loads the image, resize the image to the desired size. 
    If needed normalize the image from the range of [0 to 255] to [0 to 1].
    
    Inputs : 
        image_path : String of the path where the image file is placed.
        size       : The desired size of the image i.e the new size of the image.
        normalize  : True or False, depending on the need of normaizing the data.
    
    Output : 
        image : A Tensorflow 32Bit-Float Tensor of the Image.
    
    '''
    # Load the Image
    image = cv.imread(image_path)
    
    # Convert from BGR to RGB 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    # Resize the image to the desired size
    image = cv.resize(image, size, cv.INTER_AREA)
    
    # Convert the image to a Tensorflow 32Bit-float Tensor
    image = tf.cast(image, tf.float32)
    
    # Normalize if specified
    if normalize:
        image = image/255.
    
    # Return the processed image
    return image


def extract_label(path:str, mapping:dict) -> int:
    
    label = path.split('\\')[-2]
    label = label.split('/')[-1]

    label = mapping[label]
 
    return label 


def load_data(paths:list,  mapping:dict, size:tuple = (256,256), tf_data=False, BATCH_SIZE:int=32):
    
    '''
    This function is based on the other functions listed below : 
        1. load_image
        2. collect_label
    
    This function starts with creating a space for the images & labels to be loaded. Then it iterates through the file 
    paths and load the image & the respective label. It stores these loaded image & label. If needed, it converts this 
    numpy array data into a tensorflow data. At last, it returns the generated data.
    
    Inputs : 
        paths      : A list of all file paths of the files to be loaded.
        mapping    : This is used by the collect_label function to map the class name to a class label.
        size       : This is used by the load_image function, It represents the desired size of the image.
        tf_data    : True or False, Depending on the need of Tensorflow data.
        BATCH_SIZE : Represents the Batch Size of the Tensorflow data.
    
    Outputs : 
        all_images  : A NumPy array of all the collected images.
        all_labels  : A NUmPy array of all the collected labels.
        tf_ds       : A Tensorflow data set version of the NumPy data(Depends on tf_data). 
    
    '''
    
    # Create Space for Images and Labels
    all_images = np.empty(shape=(len(paths), *size, 3), dtype=np.float32)
    all_labels = np.empty(shape=(len(paths), 1), dtype=np.int32)
    
    # Iterate through the File Paths
    index = 0
    for path in tqdm(paths, desc="Loading"):
        
        # Load the Image
        image = load_image(image_path=path, size=size, normalize=True)
        
        # Collect the Label
        label = extract_label(path=path, mapping=mapping)
        
        # Add the images to the space
        all_images[index] = image
        all_labels[index] = label
        index += 1
    
    # Return a Tensorflow data set
    if tf_data:
        tf_ds = tfd.Dataset.from_tensor_slices((all_images, all_labels))
        tf_ds = tf_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)
        tf_ds = tf_ds.prefetch(tfd.AUTOTUNE)
        return (all_images, all_labels), tf_ds     
    
    # Return the images and Labels
    return all_images, all_labels


def show_images(images:list, labels:list, class_names:list, model:None = None, GRID:tuple = (6,10), SIZE:tuple = (30,25)) -> None:
    
    '''
    This function is responsible for creating a plot of multiple images & labels from the given dataset. The function does so by 
    plotting multiple subplots and grouping them together to make a single big plot. During the process, it randomly selects images 
    from the given data set. If a model is supplied to the function, then it will also make predictions.
    
    Inputs :
        images      : List/array of all the Images.
        labels      : List/array of all the Labels.
        class_names : List of the names of the class.
        model       : A predictive model that will make prediction on the respective image(by default its None).
        GRID        : This represents the total distribution of the plots i.e. [n_rows, n_cols]
        SIZE        : Ths size of the final figure.
    '''
    
    
    # Plotting Configuration 
    plt.figure(figsize=SIZE)
    n_rows, n_cols = GRID[0], GRID[1]
    n_images = n_rows * n_cols
    
    # Iterate through the Plot 
    for i in range(1, n_images+1):
        # Select image & label randomly
        index = np.random.randint(len(images))
        image, label = images[index], labels[index]
        title = class_names[int(label)]
        
        # If model is present, make prediction
        if model is not None:
            
            image = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
            pred = class_names[np.argmax(model.predict(image))]
            title = "True : {}\nPred : {}".format(title, pred)
            
        # Plot the Image
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(tf.squeeze(image))
        plt.title(title.title())
        plt.axis('off')
    
    # Show Final Figure
    plt.show()


# Specify the Root path
train_path = "train-images/"
valid_path = "validate-images/"
test_path = "test-images/"
Log_path = "Logs/"

# Collect the Class Names
class_names = sorted(os.listdir(train_path))
n_classes = len(class_names)


# Get all image paths
train_image_paths = collect_filepaths(train_path, class_names, shuffle=True)
valid_image_paths = collect_filepaths(valid_path, class_names, shuffle=True)
test_image_paths  = collect_filepaths(test_path,  class_names, shuffle=True)


# select a random path
idx = np.random.randint(len(train_image_paths))
path = train_image_paths[idx]


class_mapping = {name:index for index, name in enumerate(class_names)}
#extract_label(path, class_mapping)

# Load Training Data
(train_images, train_labels), train_set = load_data(paths=train_image_paths, mapping=class_mapping, tf_data=True)

# Load Validation Data
(valid_images, valid_labels), valid_set = load_data(paths=valid_image_paths, mapping=class_mapping, tf_data=True)

# Load Testing Data
(test_images, test_labels),   test_set  = load_data(paths=test_image_paths,  mapping=class_mapping, tf_data=True)


data_aug = Sequential(layers=[
    Input(shape=(256, 256, 3), name="InputLayer"),
    RandomFlip(mode='horizontal_and_vertical', name="RandomFlip"),
    RandomRotation(factor=10, fill_mode='reflect', name="RandomRotation"),
    RandomContrast(factor=0.5, name="RandomContrast")
], name="DataAugmetor")




# Give Model a Name
model_name = 'OnlyMyModelV14.2'

input_shape = (256, 256, 3)
num_classes = 5
base_model = aspiring_mobile_net(num_classes=num_classes, dropout=0.25)

# Freeze the Model Weights
base_model.trainable = True

# Final model structure
model = Sequential([
    Input(shape=input_shape),
    data_aug,
    base_model,
    #Dense(128, activation='relu'),
    #Dropout(0.2),
    Dense(num_classes, activation='softmax')
], name="m0d3l")

# Model Summary
model.summary()

# Compile Model
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),#mo¿liwa zmiana optymalizatora
    metrics=['accuracy']
)

# Model Callbacks 
callbacks = [
    EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint(model_name+".h5", monitor="val_loss", save_best_only=True)
]

# Redirect output
log_name=Log_path+model_name+"_log_"+str(date.today())+".txt"
with open(log_name,'w') as output:
    sys.stdout = output

    # Model Training
    history = model.fit(train_set, validation_data=valid_set, callbacks=callbacks, epochs=50)
    
    #print(model.history)
    # Specify the Model Path
    model_path = model_name+".h5"
    
    # Load Model
    mobilenet = load_model(model_path)
    
    # Visualize model architecture
    mobilenet.summary()

    show_images(test_images, test_labels, class_names=class_names, model=mobilenet)
    show_images(test_images, test_labels, class_names=class_names, model=mobilenet)


sys.stdout = sys.__stdout__

