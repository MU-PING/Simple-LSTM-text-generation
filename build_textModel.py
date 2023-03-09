import math
import tensorflow as tf

class TextModel():
    def __init__(self, index2vector, x_train, y_train, x_test, y_test):
        train_generator = self.train_generator()
        test_generator = self.test_generator()
        self.batch_size = 512
        self.num_classes = index2vector.shape[0]
        self.train_size = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train
        self.test_size = x_test.shape[0]
        self.x_test = x_test
        self.y_test = y_test
        train_steps_per_epoch = math.ceil(self.train_size/self.batch_size)
        test_steps_per_epoch = math.ceil(self.test_size/self.batch_size)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(index2vector.shape[0], index2vector.shape[1], weights=[index2vector], input_length=16))
        model.add(tf.keras.layers.LSTM(256, return_sequences=True))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(index2vector.shape[0], activation='softmax'))
        model.summary(line_length=100)

        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(1, name='top_1_categorical_accuracy'), 
                   tf.keras.metrics.TopKCategoricalAccuracy(5, name='top_5_categorical_accuracy')]
        
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(0.005),
                      metrics=metrics)
    
        checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='val_top_5_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
        learning_rate_function = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_top_5_categorical_accuracy', factor=0.5, patience=20, min_lr=0.0001, mode='max')

        model.fit(train_generator, steps_per_epoch=train_steps_per_epoch, 
                  epochs=300, verbose=1, callbacks=[checkpoint, learning_rate_function], 
                  validation_data=test_generator, validation_steps=test_steps_per_epoch, shuffle=True)
    
    def train_generator(self):
        index = 0
        while 1:
            start = index*self.batch_size
            end = (index+1)*self.batch_size
            yield self.x_train[start:end], tf.keras.utils.to_categorical(self.y_train[start:end], num_classes=self.num_classes)
            index = index+1 if not end>self.train_size else 0

    def test_generator(self):
        index = 0
        while 1:
            start = index*self.batch_size
            end = (index+1)*self.batch_size
            yield self.x_test[start:end], tf.keras.utils.to_categorical(self.y_test[start:end], num_classes=self.num_classes)
            index = index+1 if not end>self.test_size else 0