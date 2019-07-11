from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
import time

# CLASS LIST
# daisy = 0
# dandelion = 1
# rose = 2
# sunflower = 3
# tulip = 4

# Create Data Generators for Training and Testing
train_datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# load and iterate training dataset
train_it = train_datagen.flow_from_directory('H:/Tiliter/flowers/Training', target_size=(256,256),class_mode='categorical',\
            classes=['0','1','2','3','4'],batch_size=64)

# load and iterate test dataset
test_it = test_datagen.flow_from_directory('H:/Tiliter/flowers/Testing/', target_size=(256,256),class_mode='categorical',\
            classes=['0','1','2','3','4'],batch_size=1)

# Define CNN Model
cnn = Sequential()
# CN Layer 1
cnn.add(Conv2D(filters=8,
               kernel_size=(3,3),
               strides=(1,1),
               padding='same',
               input_shape=(256,256,3),
               data_format='channels_last'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.4))

# CN Layer 2
cnn.add(Conv2D(filters=16,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.4))

# CN Layer 3
cnn.add(Conv2D(filters=32,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))

cnn.add(Dropout(0.4))

# CN Layer 4
cnn.add(Conv2D(filters=64,
               kernel_size=(3,3),
               strides=(1,1),
               padding='valid'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2,2),
                     strides=2))
cnn.add(Flatten())

# Fully connected layer 1
cnn.add(Dense(64))
cnn.add(Activation('relu'))

cnn.add(Dropout(0.4))

# Final layer for 5 target classes
cnn.add(Dense(5))
cnn.add(Activation('softmax'))

# Performanve measure & compile network
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit network
start = time.time()
cnn.fit_generator(
    train_it,
    steps_per_epoch = 60,
    epochs=50)
end = time.time()
print('Training processing time:',(end - start)/60)

# Test accuracy on test set
start = time.time()
test_scores = cnn.evaluate_generator(test_it,528)
end = time.time()
print('Test processing time:',(end - start)/60)
print("Test set accuracy = ", test_scores[1])

cnn.save_weights('cnn_flower_prediction.h5')