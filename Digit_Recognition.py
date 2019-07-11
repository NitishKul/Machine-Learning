# Import libraries
# NOTE: Deep convolutional neural network is used to train and test handwritten digit data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
import time
# Keras framework is used for deep convolutional neural networks (CNN)

# Here, instead of extracting images into .csv file (time consuming and highly computational),
# Images are sorted on the local drive to use efficient Keras function (ImageDataGenerator and flow_from_directory)
# Images are separated based on target class names in training, validation and testing folders
# Now, ImageDataGenerator function can be directly implemented to extract images for training, validation and testing
# Images are extracted based on batch size (Efficient method while working with large datasets)
# To extract training set for training phase, validation set for cross-validation and so on,
# respective training, validation and test ImageDataGenerators are initialized
train_datagen = ImageDataGenerator( # Initialize data generator for training
    rotation_range = 40, # Image preprocessing for covering different training scenarios
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale = 1./255) # Initialize data generator for validation
test_datagen = ImageDataGenerator(rescale = 1./255) # For testing

# NOTE:
# To avoid overfitting, cross-validation set is generated from training set, furthermore,
# Pooling and dropout layers have been implemented for randomly reducing weight parameters and controlling overfitting

#load and iterate training dataset
train_it = train_datagen.flow_from_directory('H:/Tiliter/data/mnist/training/', target_size=(256,256),class_mode='categorical',\
            classes=['0','1','2','3','4','5','6','7','8','9'],batch_size=100)

# load and iterate validation dataset
val_it = validation_datagen.flow_from_directory('H:/Tiliter/data/mnist/validation/', target_size=(256,256),class_mode='categorical', \
            classes=['0','1','2','3','4','5','6','7','8','9'],batch_size=100)

# load and iterate test dataset
test_it = test_datagen.flow_from_directory('H:/Tiliter/data/mnist/testing/', target_size=(256,256),class_mode='categorical',\
            classes=['0','1','2','3','4','5','6','7','8','9'],batch_size=1)


# Define Conolutional Neural Network (CNN)
cnn = Sequential()
# Layer that has parameters (filters/ weights), is only counted as one layer,
# I.e. Pooling and dropouts are part of convolutional layer and not independent layers
# Layer 1: 8 convolutional filters for extracting low level features + Max pooling & Dropout to reduce overfitting
# Activation function: relue (Rectified linear unit)
# For first input layer, input dimensions are identical to image size (3 channels:RGB)
cnn.add(Conv2D(filters=8,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid',
               input_shape=(256,256,3),
               data_format='channels_last'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.3))

# Layer 2: 16 Filters
cnn.add(Conv2D(filters=16,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.4))

# Layer 3: 32 Filters
cnn.add(Conv2D(filters=32,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.4))

# Layer 4: 64 Filters
cnn.add(Conv2D(filters=64,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
# Output of layer 4 is flattened, to send to fully connected layers
cnn.add(Flatten())

# Fully connected layer 1
cnn.add(Dense(128))
cnn.add(Activation('relu'))

# Fully connected layer 2
cnn.add(Dense(64))
cnn.add(Activation('relu'))

cnn.add(Dropout(0.4))

# Final output layer to predict 10 target classes
cnn.add(Dense(10))
cnn.add(Activation('softmax'))

# Compile predefined model to check loss / accuracy
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model with training set and check on validation set, to avoid overfitting
start = time.time()
cnn.fit_generator(
    train_it,
    steps_per_epoch = 1000,
    epochs=10,
    validation_data=val_it,
    validation_steps = 100)
end = time.time()
print('Training processing time:',(end - start)/60)

# Test model on test set
start = time.time()
test_scores = cnn.evaluate_generator(test_it,8865) # 8865 is total number of available test samples (batch size is 1)
end = time.time()
print('Test processing time:',(end - start)/60)
print("Test set accuracy = ", test_scores[1])

# Save model weights (for future use) if it has accuracy greater than 0.9
if test_scores[1] > 0.9:
    cnn.save_weights('cnn_baseline.h5') # Default project path