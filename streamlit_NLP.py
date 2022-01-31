# -*- coding: utf-8 -*-
"""
@author: aisatux, Created on Fri Dec 24 14:12:58 2021
"""

# Core Pkgs
import streamlit as st 
import pandas as pd
import os
import time
from PIL import Image


# NLP project modules
import torch
from modules.data_loader.utils import add_padding, tensor_from_sentence, indexes_from_sentence, normalize_string, Language
from modules.data_loader.prepare_data import PrepareData
from director import Director
from main import get_parameters

#"""Code du traducteur dont on a besoin pour illustrer les propos"""
def load_params(max_length, data_dir, mode):
    #max_length, data_dir, mode = 10, "./data/fra-eng", "e2f"
    if mode == 'e2f':
        tgt_language= Language('French')
        src_language = Language('English')
    else:
        src_language = Language('French')
        tgt_language = Language('English')
    PrepareData_c = PrepareData(max_length, data_dir, mode)
    src_language, tgt_language, pairs = PrepareData_c.prepare_data()
    return src_language, tgt_language, pairs
    

def show_data(pairs):
    return pd.DataFrame(pairs[:],columns=["English","Français"]) #

def show_preprocessing(src_language, tgt_language):
    n_words = src_language.n_words
    word2count = pd.DataFrame.from_dict(src_language.word2count, orient='index',columns=["Occurrence"])
    word2index = pd.DataFrame.from_dict(src_language.word2index, orient='index',columns=["Index"])
    index2word = pd.DataFrame.from_dict(src_language.index2word, orient='index',columns=["Word"])
    return word2count, word2index, index2word, n_words

hps = get_parameters() 
hps.batch_size = 1 #mode test
max_length = hps.max_length
data_dir = hps.data_dir
mode = hps.s2t
director = Director(hps)
director.load_state_dict() #chargé le model

def test(sentence, src_language, tgt_language):
     output_words, attn_matrix, attn_applied = director.evaluate(src_language, tgt_language, sentence)
     output_sentence = ' '.join(output_words)
     return output_sentence, output_words, attn_matrix, attn_applied
        
        
#"""Code de l'application web"""
def main():
  """A Simple NLP app with Spacy-Streamlit"""
  src_language, tgt_language, pairs = load_params(max_length, data_dir, mode) #Charger l'environnement de travail
  
  #COMMUN A TOUTES LES PAGES
  menu = ["Home","State of the art","Model","Data","Processing","Training","Evaluation","Documentation"]
  choice = st.sidebar.selectbox("Menu",menu)
  st.title("SIA project 2021/2022")
  st.subheader("SOUKOUNA Aisata")
  st.subheader("")
  
  #PAGES PAR PAGES
  #""" PAGE 1"""
  if choice == "Home":
      st.subheader("Summary of the project")
      """ This project was created as part of a school project. 
      It is an educational project for which the aim is to teach other 
      people how to craft a project for machine language translation. 
      We will see on this web application the different stages of a project:
      from data processing to sentence-by-sentence translation. 
      """
      
      st.subheader("Let's see how that's work.")
      st.image("https://pytorch.org/tutorials/_images/seq2seq.png")
      """For example, we can see above, the translation of a sentence,
       in French with its equivalent in English. We observe that this 
       translation is done using an encoder and a decoder. We will see 
       in the next section how we arrived at this type of architecture 
       and how textiles can be transformed into digital data and why it 
       is necessary to carry out these transformations. """
       
      st.caption("Go to the Menu bar and select the next step you want to explore.")
  
  if choice == "State of the art":
      st.subheader("State of the art")
      st.image("https://venturebeat.com/wp-content/uploads/2018/09/natural-language-processing-e1572968977211.jpg?fit=750%2C375&strip=all")
      
      """ Machine translation relies on statistical models. They need large 
      datasets paired sentences in both the source and target language. 
      However, bilingual data is limited and there is a much larger amount 
      of monolingual data available. Monolingual data has been traditionally 
      used to train language models which improved the fluency of statistical 
      machine translation (Koehn, 2010).  
      """
      
      """ In the context of neural machine translation, which is typically 
      a neural network with an encoder/decoder architecture, there has been 
      extensive work to reach those models. During the past years, different 
      neural architectures have been proposed with the goal of improving the 
      effectiveness of machine language translation. This includes recurrent 
      networks (Sutskever et al., 2014; Bahdanau et al., 2015; Luong et al., 2015), 
      convolutional networks (Kalchbrenner et al., 2016; Gehring et al., 2017; 
      Kaiser et al., 2017), and transformer networks (Vaswani et al., 2017).
      Recent work relies on attention mechanisms where the encoder produces 
      a sequence of vectors and, for each target token, the decoder attends 
      to the most relevant part of the source through a context-dependent 
      weighted-sum of the encoder vectors (Bahdanau et al., 2015; Luong et al., 2015). 
      Attention has been refined with multi-hop attention (Gehring et al., 2017), 
      self-attention (Vaswani et al., 2017; Paulus et al., 2018), and multi-head 
      attention (Vaswani et al.,2017).
      """
      
      """
      Since 2014, the best-performing models connect the encoder and decoder 
      through an attention mechanism, avoiding recurrence and convolutions 
      entirely. Experiments on two machine translation tasks show that these 
      models are superior in quality while being more parallelizable and requiring 
      significantly less time to train. Thus, we will use a transformer architecture 
      that uses self-attention mechanisms (Vaswani et al., 2017).
      """

      st.caption("Go to the Menu bar and select the next step you want to explore.")
      
  if choice == "Model":
      st.subheader("Transformer for sequence-to-sequence translation")
      """ In 2014, Sutskever and al. proposed the sequence-to-sequence 
      architecture as a machine translation task for Natural Language 
      Processing applications. The original architecture is a Recurrent 
      Neural Network (RNN) transformer consisting of a pair of one RNN 
      encoder and decoder.
      """
      
      first_column, second_column = st.columns(2)
      with first_column :
          st.markdown("Classical encoder")
          st.image("https://pytorch.org/tutorials/_images/encoder-network.png")
      with second_column:
          st.markdown("Classical decoder")
          st.image("https://pytorch.org/tutorials/_images/decoder-network.png")
      st.latex("Figure \hspace*{0.5mm} 1 :\hspace*{1mm} Classical\hspace*{1mm} transformer")
      """The goal of an encoder is to infer a continuous space representation of 
      the source sentence, while the decoder is a neural language model conditioned 
      on the encoder output. And at the inference step, the decoder generates the 
      target sentence by left-to-right decoding of the source language.
      """
      
      st.image("https://pytorch.org/tutorials/_images/seq2seq.png")
      st.latex("Figure \hspace*{0.5mm} 2 :\hspace*{1mm} Overview \hspace*{1mm} of \hspace*{1mm} what \hspace*{1mm} the \hspace*{1mm} sequence-to-sequence \hspace*{1mm} translation \hspace*{1mm} looks \hspace*{1mm} like.")
      
      """To make it simple, the encoder encodes the input sentence, and thanks 
      to its hidden units it can save the context of the input sentence. The 
      decoder takes, subsequently, the output of the encoder as an input and 
      predicts the most probable output in the target language as we can see above.  And the parameters 
      of both models are learned jointly to maximize the likelihood of the target 
      sentences given the corresponding source sentences from bilingual data.
      """
      
      st.subheader("Mathematics behind the machine translation task")
      """The machine translation task is in reality the modeling of the conditional 
      probability P(y|x) where x and y refer to the input and output sentence, 
      respectively. The purpose of this technology is to, for a given sentence 
      x in language X, find the best sentence y in another language Y. Mathematically, 
      this can be rewritten as (Equation. 1), that is we want to find the output 
      sentence y that maximizes the conditional probability P(y|x), which can be 
      rewritten using Bayes’ Rule as P(x|y)P(y).
      """
      st.image("https://miro.medium.com/max/490/1*ioYIwuqWWyrJCLW8Kve6BA.png")
      st.latex("Equation \hspace*{0.5mm} 1 :\hspace*{1mm} Maximum\hspace*{1mm} a \hspace*{1mm} posteriori")
      
      """Here P(x|y) models the translation model, i.e. how words and sentences are
      translated, and P(y) models the target language model which is Y. And thanks 
      to the search algorithm, such as the Greedy or the Beam algorithm, we 
      look for the most likely sentence.
      """
      
      st.markdown("For instance, P(“I am studying”) > P(“I studying am”)?")
      
      st.subheader("Search algorithm")
      """Search algorithms are used in each machine translation as a final 
      decision-making layer to choose the optimal output given probabilities 
      of target variables. There are two commonly used searching algorithms, 
      it's about Greedy and Beam search algorithms. 
      """
      st.image("https://miro.medium.com/max/1400/1*NOERSs9amkoYnCHQE5s3DA.png")
      st.latex("Figure \hspace*{0.5mm} 3 :\hspace*{1mm} Overview \hspace*{1mm} of \hspace*{1mm} what \hspace*{1mm} the \hspace*{1mm} greedy \hspace*{1mm} search \hspace*{1mm} algorithm \hspace*{1mm} works.")
      
      """The Greedy search method will simply take the highest probability 
      word at each position in the sequence and predict that in the output 
      sequence. Choosing just one candidate at a step might be optimal at 
      the current spot in the sequence, but as we move through the rest of 
      the sentence, it might turn out to be not optimal. This is the main 
      disadvantage of the Greedy method, the optimal output doesn't take 
      into account the sentence context. This is not as bad as its looks 
      because this is the purpose of the transformer's encoder.
      """
      
      """On the other hand, the beam search algorithm selects multiple tokens 
      at each step, giving multiple probable outputs to a given input sentence, 
      based on conditional probability. The algorithm takes as a parameter, 
      the beam-width, the number of the best alternatives at each step of 
      decoding. The main disadvantage of this search algorithm is that it 
      needs a lot of memory resources and took time to process.
      """
      
      """The beam search method is the preferred search strategy for the 
      sequence to sequence algorithms, but we will evaluate our model, 
      thanks to the Greedy search strategy and we will see that it is 
      enough for our task.
      """
      
      st.subheader("The chosen model")
      
      """In 2017, Vaswani and al proposed the sequence-to-sequence architecture with 
      self-attention mechanisms for machine translation task for Natural Language 
      Processing applications. This approach have improved the performance of machine 
      translation models, compared to classical thanks to the attention mechanism. Thus, 
      we choose the model described below as working model. """
      
      first_column, second_column = st.columns(2)
      with first_column :
          st.markdown("Classical encoder")
          st.image("https://pytorch.org/tutorials/_images/encoder-network.png")
      with second_column:
          st.markdown("Attention decoder")
          st.image("https://pytorch.org/tutorials/_images/attention-decoder-network.png")
      st.latex("Figure \hspace*{0.5mm} 4 :\hspace*{1mm} Working \hspace*{1mm} transformer")
      
      """We can see below what is self-attention mechanism and how it works."""

      st.subheader("Self-attention mechanism")
      
      """The attention mechanism seek to know what part of the input should 
      we focus on. For translation from English to French, we want to know 
      how relevant a word in the english sentence is relevant to other words 
      in the same sentence. This is represented in the attention vector. 
      For every word we can generate an attention vector that captures the 
      contextual relationship between words in a sentence.
      """
      
      st.image("https://theaisummer.com/static/4022cf02281d234e0e85fa44ad08b4e2/9f933/self-attention-probability-score-matrix.png")
      st.latex("Figure \hspace*{0.5mm} 5 :\hspace*{1mm} Overview \hspace*{1mm} of \hspace*{1mm} what \hspace*{1mm} the \hspace*{1mm} attention \hspace*{1mm} matrix \hspace*{1mm} looks \hspace*{1mm} like.")
      """For example, we can see above how each word in the same sentence are 
      correlated with each other. We observe that the coefficients located 
      on the diagonal have the highest values because they are the same words. 
      Note that these values are not equal to 1. We also observe that, although 
      the coefficients outside the diagonal are rather low, they are not zero and 
      show the inter-word correlation of a sentence. 
      """
      
      """We will see in the next section what the data looks like and 
       how we transform it to feed into the working model. """
       
      st.caption("Go to the Menu bar and select the next step you want to explore.")
      
  if choice == "Data":
      st.subheader("Overview of the data")
      data = show_data(pairs)
      st.markdown(f"We are working with {len(data)} bilingual pairs of data in which one data refers to a sentence and its translation.")
      st.markdown("Here is an overview of our database :")
      data
      
      st.subheader("Tokenization")
      """ The transformer revolution started with the willingness of no dependencies 
      between each hidden state. To have it, we split the input sentence into 
      words, and this step is called tokenization. This is the first step 
      before we feed the input into the model. The second step is to build 
      word embedding and this is the first layer of an encoder. 
      """
      
      st.image("https://theaisummer.com/static/c9a851690a62f1faaf054430ca35ab20/c7dcc/tokenization.png ")
     
      """ The tokenization step transforms the sentence into a set and the 
      notion of order is lost. And because a neural network certainly cannot 
      understand any order in a set, we use to give as input, at each step 
      of encoding/decoding, the previous word in the set. The limit of this 
      is that we cannot have many times the same words in the input sentence.
      """
      
      st.subheader("About the database")
      st.caption("Working configuration:")
      st.markdown(f"Sentences should have a maximum length of {max_length} words/signs."
              "\nWe can translate from English to French, or vice versa."
              "\nBut let’s focus on the English to French translation.")

      
      word2count, word2index, index2word, n_words = show_preprocessing(src_language, tgt_language)
      
      st.caption(" First step : Counting words occurence in the source language")
      word2count #database
      st.markdown(f"There are {n_words} different words in the database.")
      
      st.caption(" Second step : Transform words into numbers")
      left_column, right_column = st.columns(2)
      with left_column :
          st.markdown("Encodage code : word to number")
          word2index #database
      with right_column :
          st.markdown("Decodage code : number to word")
          index2word #database
      st.markdown("You can look above if the encoding's code and decoding's code match.")
      st.caption("Go to the Menu bar and select the next step you want to explore.")
      
      
  if choice == "Processing":
      st.subheader("Processing")
      st.caption("Working configuration:")
      st.markdown(f" * Sentences should have a maximum length of {max_length} words/signs.")
      if mode == 'e2f':
          st.markdown(" * We translate from English to French.")
      else:
          st.markdown(" * We translate from French to English.")
      st.markdown("")
      
      st.caption("Let's test the preprocessing!")
      sentence = st.text_area("Text to translate:","He held out a helping hand to the poor")
      
       
      if st.button("Processing"):
          st.markdown("1. Normalization step")
          sentence = normalize_string(sentence)
          st.code(sentence)
      
          st.markdown("2. Encodage step")
          try:
              input_index = indexes_from_sentence(src_language, sentence)
              words = [word for word in sentence.split(' ')]
              df = pd.DataFrame(input_index).T
              df.columns = words
              df
          except Exception:
              st.code("Warning: if there is an error type such 'KeyError: 'text''\n"
      "this means that the word 'text' isn't in the input language database.\n"
      "Change the sentence to translate!")
              st.code("Warning: if there is an error type such 'Duplicate column names found'\n"
      "this means that you cannot have two same words in a one sentence.\n"
      "Change the sentence to translate!")
          
          st.markdown("3. Shape normalization step")
          input_tensor = tensor_from_sentence(src_language,  str(sentence))
          words_em = ['CLS']+[word for word in sentence.split(' ')]+['SEP']
          df = pd.DataFrame(input_tensor).T
          df.columns = words_em
          df
          
        
          try:
              st.markdown("4. Length normalization")
              input_tensor, _ = add_padding(input_tensor, max_length)
              n_pad = max_length - len(words_em)
              words_pad = ['CLS']+[word for word in sentence.split(' ')]+['SEP']+['PAD']
              
              if n_pad>1:
                  st.markdown("We have to add padding to normalize the sentence for our model.")
              
              st.markdown(f"The length of our sentence is {len(words_em)}, so we have added {n_pad} "
                      f"padding indexes to our initial sentence to have {max_length}, the mandatory length.")
              st.markdown("See the effects of the padding below.")
              
              df = pd.DataFrame(input_tensor[:-n_pad+1]).T
              df.columns = words_pad
              df
              d = torch.tensor(input_tensor, dtype=torch.long)
              d
          except Exception:
              df = pd.DataFrame(input_tensor).T
              df.columns = words_pad
              df
              d = torch.tensor(input_tensor, dtype=torch.long)
              d
              pass
          st.markdown("5. Attention matrix")
          prediction, output_words, attn_matrix, attn_applied = test(sentence, src_language, tgt_language)
          st.caption("Self-attention map")
          path = director.plt_heatmap(attn_applied, sentence, sentence, True)
          st.image(Image.open(path))
          st.caption("Go to the Menu bar and select the next step you want to explore.")
      
        
  if choice == "Training":
      st.header("Training")
      
      st.subheader("The chosen model")
      
      """In 2017, Vaswani and al proposed the sequence-to-sequence architecture with 
      self-attention mechanisms for machine translation task for Natural Language 
      Processing applications. This approach have improved the performance of machine 
      translation models, compared to classical thanks to the attention mechanism. Thus, 
      we choose the model described below as working model. """
      
      first_column, second_column = st.columns(2)
      
      if st.button("Working transformer"):
          with first_column :
              st.markdown("Classical encoder")
              st.image("https://pytorch.org/tutorials/_images/encoder-network.png")
          with second_column:
              st.markdown("Attention decoder")
              st.image("https://pytorch.org/tutorials/_images/attention-decoder-network.png")
              
      st.subheader("Word Embedding")
      """
      As we can see above, the first layer of the encoder is a word embbeding’s 
      layer. In general, an embedding is a representation of a symbol (word, 
      character, sentence) in a distributed low-dimensional space of 
      continuous-valued vectors. Words are not discrete symbols. They are 
      strongly correlated with each other. That’s why when we project them 
      in a continuous euclidean space we can find associations between them.
      """

      st.subheader("Working configuration")
      """ There are many parameters and hyperparameters to set in order to 
      achieve optimal results. Here is the configuration an overview of 
      parameters to set to train the model."""
      if mode == 'e2f':
          lang = "English-to-French"
      else: 
          lang = "French-to-English"
          
      st.caption("Training configuration:")
      st.markdown(f" * translation mode : {lang} \n * the proportion of the validation set  : 0.2"
                  "\n * the proportion of the train set : 0.8 \n * maximal length of input sentence : 12"
                  "\n  * batch size : 64 \n * learning rate : 0.005 \n * number of epochs : 100")
      
      
      st.subheader("Training results")
      
      """ First let’s compare the two different models, 
      the classical transformer and the transformer with attention mechanism. 
      We set to both the same working configuration in order to compare it."""
      
      first_column, second_column = st.columns(2)
      with first_column:
          st.caption(" Attention model with lr = 0.005, dropout = 0.1, epoch = 50 ")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_attention_0.285.png')))
      with second_column:
          st.caption(" Classical model with lr = 0.005, dropout = 0.1, epoch = 50 ")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_vanilla_0.456.png')))
          
      """Here, the loss of the classical model reach 0.456 at the end of the 50 epochs, while
      the loss of the attention model reach 0.285. We observe that, with this fixed setup, 
      the attention model is better than the classical model."""
      
      first_column, second_column = st.columns(2)
      with first_column:
          st.caption(" Attention model with lr = 0.01, dropout = 0.1, epoch = 50")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_attention_1.786_dropout_0.1_LR_0.01.png')))
      with second_column:
          st.caption(" Classical model with lr = 0.01, dropout = 0.1, epoch = 50")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_vanilla_dropout_0.1_LR_0.01.png')))
          
      """ As seen before, the attention model is again better than the classical, so we 
      will focus only on attention model to seek the optimal configuration. Moreover, 
      we observe that the train is not stable with a learning rate lr equals to 0.01. It was better before. """
      
      """ Let's see the result of attention model with others setting configuration."""
      
      first_column, second_column = st.columns(2)
      with first_column:
          st.caption(" Attention model with lr = 0.005, dropout = 0.5, epoch = 30 ")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_lr=0.005.png')))
          
      with second_column:
          st.caption(" Attention model with with lr = 0.001,, dropout = 0.5, epoch = 20 ")
          st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_lr=0.001.png')))
      
      """ We observe that the train is not stable with a dropout equals to 0.5. 
      It was better with a dropout equals to 0.1. """
      
      """ We observe the train is stable with a learning rate lr equals to 0.001 too. 
       Thus, with a learning rate lr under 0.005 the training seems to be stable. 
       But, the lower is the learning rate, the slower is the train. 
       Thus, we will work with the attention model with the first configuration,
       which is described below."""
      
      st.subheader("Selected configuration:")
      st.markdown(f" * translation mode : {lang} \n * the proportion of the validation set  : 0.2"
                  "\n * the proportion of the train set : 0.8 \n * maximal length of input sentence : 12"
                  "\n  * batch size : 64 \n * learning rate : 0.005 \n * dropout : 0.1 \n " 
                  "* number of epochs : 50")
      st.image(Image.open(os.path.join(os.getcwd(),'photos/outputs_loss_attention_0.285.png')))

      st.caption("Go to the Menu bar and select the next step you want to explore.")
  
  if choice == "Evaluation":
      st.header("Evaluation of the model")
      st.caption("Working configuration:")
      st.markdown(f" * Sentences should have a maximum length of {max_length} words/signs.")
      st.markdown(" * We translate from French to English.")
      
      st.caption("Let's test the preprocessing!")
      sentence = st.text_area("Text to translate:","what time did he get there?")
        
      st.sidebar.title("Controls")
      start = st.sidebar.button("Start")
      stop = st.sidebar.button("Stop")
      
      
      try:
          if start:
              prediction, output_words, attn_matrix, attn_applied = test(sentence, src_language, tgt_language)
              latest_iteration = st.empty()
              bar = st.progress(0)
              for i in range(100):
                  latest_iteration.text('Loading') #
                  bar.progress(i + 1)
                  time.sleep(0.005)
              st.text_area("Translation:", prediction)
              
              st.caption("Self-attention map")
              path = director.plt_heatmap(attn_applied, sentence, sentence, True)
              st.image(Image.open(path))
              
              st.caption("Attention map")
              path = director.plt_heatmap(attn_matrix, sentence, output_words)
              st.image(Image.open(path))
          if stop:
            pass
      except Exception:
              """ Delete '?' from the sentence you wish to translate and repeat !"""
              pass
      
      
    
      st.caption("Go to the Menu bar and select the next step you want to explore.")

      
if __name__ == '__main__':
	main()