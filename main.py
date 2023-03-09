import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
Must be in the same file as tensorflow and before tensorflow is imported
"""
import sys
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from build_word2vec import Preprocessing
from build_textModel import TextModel

print("Environment confirmation:", sys.executable)
print("TensorFlow version:", tf.__version__)
print("----------------------------------------------------------------------------------------------------------------------------------------")
print("Check devices available to tensorFlow")
print(device_lib.list_local_devices())
print("----------------------------------------------------------------------------------------------------------------------------------------")
print("Hardware devices")
print(tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices(device_type='GPU'))
print("----------------------------------------------------------------------------------------------------------------------------------------")
print("Usable CPU or GPU (visible_devices) by tensorflow")
print(tf.config.get_visible_devices())
print("----------------------------------------------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    step = 16

    # preprocess data
    preprocessing = Preprocessing(step)

    # train model
    #textModel = TextModel(preprocessing.index2vector, preprocessing.x_train, preprocessing.y_train, preprocessing.x_test, preprocessing.y_test)

    # inference model 
    model = tf.keras.models.load_model('model.h5')
    templates = ["template1.txt"]

    for template in templates:
        with open("Fabricated-Fairy-Tales/"+template, 'r',encoding="utf-8") as f:
            text = f.read()
            word_tokenize = nltk.word_tokenize(text)

        template_data = list(map(preprocessing.word2index.get, word_tokenize[-step:]))

        all_index = [i for i in range(preprocessing.index2vector.shape[0])] 

        for _ in range(300):

            # model predict
            nextword = model(np.asarray([template_data]))

            # normalize 
            p = np.array(nextword[0, -1, :])
            p /= p.sum()

            nextword = int(np.random.choice(all_index, 1, p=p))
            template_data.pop(0) #delete first word
            template_data.append(nextword)
            
            nextword = preprocessing.index2word[nextword]
            apostrophe = [',', ':', '.', '“', '”', '‘', '’', ';', '?']
            if nextword in apostrophe:
                text = text + nextword
            else:
                text = text + " " + nextword
            
        with open("Fabricated-Fairy-Tales/Fabricated-Fairy-Tales_"+template, 'w') as f:
            f.write(text)
