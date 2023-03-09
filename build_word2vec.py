import os
import nltk
import numpy as np
import tensorflow as tf
nltk.download('punkt');
from gensim.models.word2vec import Word2Vec

class Preprocessing():
    def __init__(self, step):
        
        #為個別txt檔之word_tokenize
        story_word = [] 
        sentences_word = []
        
        for file in os.listdir("Famous-Fairy-Tales/"):
            article_word = []
            with open("Famous-Fairy-Tales/"+file, 'r', encoding="utf-8") as f:        
                for paragraph in f.read().split("\n"):                                   # 建議使用 '\n' 或 'sent_tokenize' 分割句子，整篇文章下去分詞(word_tokenize)可能會出現錯誤
                    for sentence in nltk.tokenize.sent_tokenize(paragraph):
                        sentence_word = nltk.tokenize.word_tokenize(sentence)            # 為甚麼不能用split("")而用word_tokenize 解：https://reurl.cc/Q32y1p
                        sentences_word.append(sentence_word)
                        article_word.extend(sentence_word)
            story_word.append(article_word)

        # Word2Vec 的 input 可以是 1D list 或 2D list（亦可直接使用 numpy 或 pandas 的結構）
        # 1D list：[ '我是智障', '你媽超瘦' ] → 會被 Word2Vec 以單個字元分詞成 [[ '我', '是', '智', '障' ], [ '你', '媽', '超', '瘦' ]] 來訓練
        # 2D list：[[ '我', '是', '智障' ], [ '你媽', '超瘦' ]] → Word2Vec 會直接使用我們分好的詞來訓練
        word2Vec = Word2Vec(sentences_word, min_count=1, vector_size=32, window=5, epochs=50)
        
        # word2vec 使用 Cosine Similarity 來計算兩個詞的相似性．這是一個 -1 到 1 的數值，如果兩個詞完全一樣就是 1
        # Second, if in fact the model is getting worse at an external evaluation – like whether the most_similar() results match human estimations – with more training, 
        # then that's often an indication the overfitting is occurring. That is, the (possibly oversized) model is memorizing features of the (likely undersized) training data, 
        # and thus getting better at its internal optimization goals in ways that no longer generalize to the larger world of interest.
        print("\nThe word is most similar with 'day':", word2Vec.wv.most_similar('day'), "\n")
        
        # index2word [Word2vec_model].wv.index_to_key
        # word2index 利用 index2word 來算出
        # index2vector [Word2vec_model].wv.vectors
        self.index2word = word2Vec.wv.index_to_key   
        self.word2index = {token: token_index for token_index, token in enumerate(word2Vec.wv.index_to_key)}                            
        self.index2vector = word2Vec.wv.vectors

        # training data parameters 
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []

        # word2index and window spilt sentence
        story_index = [list(map(self.word2index.get, x)) for x in story_word] # map word to index
        print("The first ten 'Word' of the first story: ", story_word[0][0:10])
        print("The first ten 'Index' of the first story: ", story_index[0][0:10], "\n")

        # train
        for article_word in story_index[:-1]:
            index = 0
            size = len(article_word)
            x = article_word[:-1]
            y = article_word[1:] 

            while index+step < size:
                self.x_train.append(x[index:index+step])
                self.y_train.append(y[index:index+step])
                index += 1

        # test
        article_word = story_index[-1]
        index = 0
        size = len(article_word)
        x = article_word[:-1]
        y = article_word[1:] 
        while index+step < size:
            self.x_test.append(x[index:index+step])
            self.y_test.append(y[index:index+step])
            index += 1

        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)
        self.x_test = np.asarray(self.x_test)
        self.y_test = np.asarray(self.y_test)

        print("x_train.shape:", self.x_train.shape)
        print("y_train.shape:", self.y_train.shape)
        print("x_test.shape:", self.x_test.shape)
        print("y_test.shape:", self.y_test.shape, "\n")