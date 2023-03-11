# Simple-LSTM-text-generation
## Pretrained Embedding Layer by Word2Vec
* Word2vec is a technique for natural language processing (NLP) published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.
* Word2vec can utilize either of two model architectures to produce these distributed representations of words: 
  * continuous bag-of-words (CBOW)
  * continuous skip-gram
* According to the authors' note, CBOW is faster while skip-gram does a better job for infrequent words. 
![image](https://user-images.githubusercontent.com/93152909/224398505-3be5ee54-db65-4088-89d2-a2b44a617a3d.png)  
Image: https://kavita-ganesan.com/comparison-between-cbow-skipgram-subword/#.ZAt5bHZByHs

## LSTM Text Generation
### LSTM Model
![image](https://user-images.githubusercontent.com/93152909/224502283-2bbd8ec7-73e6-462b-a97f-3ab3afffe9e6.png)

### Input and Output Data
![image](https://user-images.githubusercontent.com/93152909/224502685-38db9c8f-0a9a-46ee-82e6-2a221c356427.png)

## Results
> Results are ungrammatical or same as training data

There was once a woman who wished very much to have a little child, but she could not obtain her wish. At last she went to a fairy, and said,‘ but if you should become it is?’ The blood fell crying, it said:‘ What lay just as fine men.’ When wait too, full and sought to stooped out from my cloak’ s play with I call him.’ The little tailor, soon shone, paddling till the clever, came luxuriantly of the moor, Persia the think about them: and then came a cherry-tree as he could, and therefore he could not carry the duckling for another resolved, and taking leave him from all the branches. The fisherman was surprised at him all a large year. He took him back on her bread looking very great that could not fall out. He would have plenty of him. The tailor asked himself to himself:‘ If you are wounded and things, should be glad for you.”“ persons your finger.’ Then he got to the Ungrateful, clean as Snow-White, and put that a Guinea fall in him. The little tailor called the crow:‘ Here is a unicorn African matter.’‘ However,’ said he,‘ it is hung together.” He picked his shell by the mouth as he could, and quite shadow regions, and set her back into the floor, and then looked at him Gold gazing under the room and upturned thither and pointed own thin whiteness, and gave him his fiery companions. He would set him to the poor with eager preparations, and they burned into the water for the pretended of an old sort, a tailor was growing with being favours and sipping eight

* Why
  * Overfitting due to small or little data
 
* How to solve
  * Collect more training data
  * Cleaner pre-processing
  * Use "Pretrained NLP" model
  
