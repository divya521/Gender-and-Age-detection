#data augmentation
trainDataGen = ImageDataGenerator(horizontal_flip=True,
                                 brightness_range=[0.2,1.0],
                                 height_shift_range=0.2,
                                 width_shift_range= 1,
                                 rotation_range=5,
                                 rescale=1.0/255,
                                 zca_epsilon=1e-06,
                                 fill_mode = 'nearest')

testDataGen = ImageDataGenerator(rescale=1.0/255)

trainGenerator = trainDataGen.flow_from_directory(os.path.join("ASplitted_Dataset","ASplitted_Dataset", "Train"),
                                                 classes=[str(Class) for Class in range(8)],
                                                 class_mode="categorical",
                                                 color_mode="rgb",
                                                 target_size=(200,200),
                                                 batch_size= 32,
                                                 shuffle=True,)

validationGenerator = testDataGen.flow_from_directory(os.path.join("ASplitted_Dataset","ASplitted_Dataset", "Validation"),
                                                 classes=[str(Class) for Class in range(8)],
                                                 class_mode="categorical",
                                                 color_mode="rgb",
                                                 target_size=(200,200),
                                                 batch_size= 32,
                                                 shuffle=True,)

                                       
  model = Sequential()
    

model.add(Conv2D(128, kernel_size=3, input_shape=(200,200,3),activation='relu'))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(Conv2D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(Conv2D(128, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=3,activation='relu'))
model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
model.add(BatchNormalization())

# model.add(Conv2D(256, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))

# model.add(Conv2D(512, kernel_size=3, input_shape=(200,200,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())

# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# # model.add(Conv2D(512, kernel_size=3, activation='relu'))
# # model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))





# model.add(Conv2D(256, kernel_size=3, activation='relu'))
# model.add(Conv2D(256, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())

# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(Conv2D(512, kernel_size=3, activation='relu'))
# model.add(MaxPooling2D(pool_size=3, strides=2))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(8, activation='sigmoid'))



model.compile(loss="categorical_crossentropy", optimizer = Adam(lr = 1e-3, decay = 1e-5), metrics=['accuracy'])
    
callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                              patience = 7, min_lr = 1e-5),
             EarlyStopping(patience = 9, # Patience should be larger than the one in ReduceLROnPlateau
                          min_delta = 1e-5),
             CSVLogger(r"/content/drive/MyDrive/AModel/training.log", append = True),
             ModelCheckpoint(r'/content/drive/MyDrive/AModel/backup_last_model.hdf5'),
             ModelCheckpoint(r'/content/drive/MyDrive/AModel/best_val_acc.hdf5', monitor = 'val_accuracy', mode = 'max', save_best_only = True),
             ModelCheckpoint(r'/content/drive/MyDrive/AModel/best_val_loss.hdf5', monitor = 'val_loss', mode = 'min', save_best_only = True)]

model.fit(trainGenerator, epochs = 30, validation_data = validationGenerator, callbacks = callbacks)

model = load_model(r'/content/drive/MyDrive/AModel/best_val_loss.hdf5')
loss, acc = model.evaluate(validationGenerator)

print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))    
