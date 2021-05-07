import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense


trainDataGen = ImageDataGenerator(horizontal_flip=True,
                                 brightness_range=[0.2,1.0],
                                 channel_shift_range=0.7,
                                 rotation_range=50,
                                 rescale=1.0/255,
                                 zca_epsilon=1e-06,
                                 fill_mode = 'nearest')

testDataGen = ImageDataGenerator(rescale=1.0/255)

trainGenerator = trainDataGen.flow_from_directory(os.path.join("Splitted_Dataset","Splitted_Dataset", "Train"),
                                                 classes=[str(Class) for Class in range(2)],
                                                 class_mode="binary",
                                                 color_mode="rgb",
                                                 target_size=(227,227),
                                                 batch_size= 32,
                                                 shuffle=True,)

validationGenerator = testDataGen.flow_from_directory(os.path.join("Splitted_Dataset","Splitted_Dataset", "Validation"),
                                                 classes=[str(Class) for Class in range(2)],
                                                 class_mode="binary",
                                                 color_mode="rgb",
                                                 target_size=(227,227),
                                                 batch_size= 32,
                                                 shuffle=True,)



model = tf.keras.Sequential()


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(227,227,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


model.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    
callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                              patience = 7, min_lr = 1e-5),
             EarlyStopping(patience = 9, # Patience should be larger than the one in ReduceLROnPlateau
                          min_delta = 1e-5),
             CSVLogger(r"/content/drive/MyDrive/Model/training.log", append = True),
             ModelCheckpoint(r'/content/drive/MyDrive/Model/backup_last_model.hdf5'),
             ModelCheckpoint(r'/content/drive/MyDrive/Model/best_val_acc.hdf5', monitor = 'val_accuracy', mode = 'max', save_best_only = True),
             ModelCheckpoint(r'/content/drive/MyDrive/Model/best_val_loss.hdf5', monitor = 'val_loss', mode = 'min', save_best_only = True)]

model.fit(trainGenerator, epochs = 25, validation_data = validationGenerator, callbacks = callbacks)

model = load_model(r'/content/drive/MyDrive/Model/best_val_loss.hdf5')
loss, acc = model.evaluate(validationGenerator)

print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))    
