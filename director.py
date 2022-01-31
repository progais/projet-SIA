import os
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from modules.encoder.encoder import EncoderRNN
from modules.data_loader.data_loader import DataLoaderProducer
from modules.data_loader.prepare_data import PrepareData
from modules.data_loader.utils import SpecialToken
from modules.decoder.vanilla import DecoderRNN
from modules.decoder.attention import AttnDecoderRNN
from modules.data_loader.utils import tensor_from_sentence, add_padding, normalize_string

from utils import set_logger, log, RunningAverage


class Director:
    def __init__(self, hps):
        os.environ['CUDA_VISIBLE_DEVICES'] = hps.gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hps = hps

        self.dl_producer = DataLoaderProducer(max_length=hps.max_length, data_dir=hps.data_dir, mode=hps.s2t)
        self.pdata_class = PrepareData(max_length=hps.max_length, data_dir=hps.data_dir, mode=hps.s2t)
        self.src_language, self.tgt_language, self.train_pairs, self.val_pairs = self.dl_producer.prepare_data_loader(batch_size=hps.batch_size, val_split=hps.val_split)
        self.encoder = EncoderRNN(batch_size=hps.batch_size, input_vocabulary_size=self.src_language.n_words,
                                  hidden_size=hps.hidden_size).to(self.device)
        self.decoder = self.get_decoder().to(self.device)
        self.global_step = 0

    def get_decoder(self):
        """ Choix du type de decoder"""
        if self.hps.decoder_type == 'attention':
            return AttnDecoderRNN(output_vocabulary_size=self.tgt_language.n_words, batch_size=self.hps.batch_size,
                                  hidden_size=self.hps.hidden_size, dropout_rate=self.hps.dropout_rate,
                                  max_length=self.hps.max_length)
        elif self.hps.decoder_type == 'vanilla': #decodeur classique RNN
            return DecoderRNN(output_vocabulary_size=self.tgt_language.n_words, batch_size=self.hps.batch_size,
                              hidden_size=self.hps.hidden_size)
        else:
            raise ValueError('decoder type is illegal')

    """ Configuration des répertoire de travail"""
    @property
    def log_dir(self):
        log_dir = os.path.join(self.hps.model_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    @property
    def ckpt_dir(self):
        ckpt_dir = os.path.join(self.hps.model_dir, 'ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @property
    def summ_dir(self):
        summ_dir = os.path.join(self.hps.model_dir, 'summary')
        os.makedirs(summ_dir, exist_ok=True)
        return summ_dir

    @property
    def heat_map_dir(self):
        heat_map_dir = os.path.join(self.hps.model_dir, 'heat_map')
        os.makedirs(heat_map_dir, exist_ok=True)
        return heat_map_dir
    

    def train(self):
        """ Fonction d'entrainement du modèle pour plusieurs epochs. 
        On y enrgistre les cnditions d'entrainements."""
        
        set_logger(os.path.join(self.log_dir, 'train.log'), terminal=False)

        epochs = self.hps.num_epochs
        print_every = self.hps.print_every
        log_every = self.hps.log_summary_every
        lr = self.hps.learning_rate

        loss_avg = RunningAverage() 
        loss_avg_val = RunningAverage()
        summary_writer = SummaryWriter(log_dir=self.summ_dir)
        current_best_loss = 1e3

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)


        criterion = nn.NLLLoss(reduce=False)

        if self.hps.resume:
            log('- load ckpts...')
            self.load_state_dict()
            
        self.train_loss, self.val_loss = [],[]

        for epoch in trange(epochs, desc='epochs'):
            loss_avg.reset()
            loss_avg_val.reset()
            
            with tqdm(total=len(self.train_pairs)) as progress_bar:
                for language_pair, mask_pair in self.train_pairs:
                    language_pair, mask_pair = language_pair.to(self.device), mask_pair.to(self.device)
                    loss = self.train_single(language_pair, mask_pair, encoder_optimizer,
                                             decoder_optimizer, criterion)
                    loss_avg.update(loss.item())
                    self.global_step += 1
                    if self.global_step % log_every == 0:
                        summary_writer.add_scalar('loss_value', loss, global_step=self.global_step)
                    if self.global_step % print_every == 0:
                        log('global training step: {}, loss average: {:.3f}'.format(self.global_step, loss_avg()))
                    
                    progress_bar.set_postfix(loss_avg=loss_avg())
                    progress_bar.update()
            
            with tqdm(total=len(self.val_pairs)) as progress_bar:
                for language_pair_val, mask_pair_val in self.val_pairs:
                    language_pair_val, mask_pair_val = language_pair_val.to(self.device), mask_pair_val.to(self.device)
                    loss = self.train_single(language_pair_val, mask_pair_val, encoder_optimizer,
                                             decoder_optimizer, criterion)
                    loss_avg_val.update(loss.item())
                    
                    progress_bar.set_postfix(loss_avg_val=loss_avg_val())
                    progress_bar.update()
            log('validation step : loss average: {:.3f}'.format(loss_avg_val()))
            
            self.train_loss.append(loss_avg())
            self.val_loss.append(loss_avg_val())
            
            if loss_avg_val() < current_best_loss:
                log('new best loss average found, saving modules...')
                current_best_loss = loss_avg_val()
                state = {
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'global_step': self.global_step,
                    'epoch': epoch,
                    'loss_avg': loss_avg_val()
                }
                torch.save(state, os.path.join(self.ckpt_dir, 'best.pth.tar'))
                
        self.save_plots(self.train_loss,self.val_loss)
        

    def train_single(self,
                     language_pair,
                     mask_pair,
                     encoder_optimizer: optim.Optimizer,
                     decoder_optimizer: optim.Optimizer,
                     criterion):
        """ Fonction d'entrainement du modèle pour 1 epoch"""
        input_tensors = language_pair[:, 0, :].t()
        target_tensors = language_pair[:, 1, :].t()
        target_masks = mask_pair[:, 1, :].t()

        encoder_init_hidden = self.encoder.init_hidden(self.device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        sentence_length = self.hps.max_length

        loss = 0
        
        encoder_output, encoder_final_hidden = self.encoder(input_tensors, encoder_init_hidden)

        decoder_input = torch.tensor([SpecialToken.CLS_token] * self.hps.batch_size, device=self.device)

        decoder_hidden = encoder_final_hidden

        use_teacher_forcing = True if random.random() < self.hps.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for di in range(sentence_length):
                decoder_output, decoder_hidden, _, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                loss += torch.sum(criterion(decoder_output, target_tensors[di]) * target_masks[di])
                decoder_input = target_tensors[di]
        else:
            for di in range(sentence_length):
                decoder_output, decoder_hidden, _, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                topv, topi = decoder_output.topk(1) #topv <- valeur, topi <- indice
                #topk(1) --> renvoie le dernier élément de decoder_output
                decoder_input = topi.squeeze().detach()
                loss += torch.sum(criterion(decoder_output, target_tensors[di]) * target_masks[di])
                

        loss = loss / target_masks.sum()
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss

    def load_state_dict(self):
        """ Chargement du modèle et de ses conditions d'entrainements"""
        ckpt_path = os.path.join(self.hps.model_dir, 'ckpts', 'best.pth.tar')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('- no ckpt file found')
        state_dict = torch.load(ckpt_path)
        self.global_step = state_dict['global_step']
        self.encoder.load_state_dict(state_dict['encoder'])
        self.decoder.load_state_dict(state_dict['decoder'])
        log('- load ckpts from global step {}'.format(self.global_step))

    def test(self):
        """ Test sur des nouvelles données"""
        set_logger(os.path.join(self.log_dir, 'test.log'), terminal=False)
        self.load_state_dict()
        src_language, tgt_language, pairs = self.pdata_class.prepare_data()
        
        log('- greedy search')

        for i in range(self.hps.sample_num):
            pair = random.choice(pairs)
            log('> ' + pair[0])
            log('= ' + pair[1])
            print("pair[0]",pair[0])
            output_words, _, _ = self.evaluate(src_language, tgt_language, pair[0])
            output_sentence = ' '.join(output_words)
            log('< ' + output_sentence)
            log('')
 
        
    def evaluate(self, input_lang, output_lang, sentence):
        """ Fonction de prédiction: traduction de la phrase d'entree dans la langue souhaité"""
        sentence = normalize_string(sentence)
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_tensor, _ = add_padding(input_tensor, self.hps.max_length)
        with torch.no_grad():
            input_tensor = torch.tensor(input_tensor, dtype=torch.long).to(self.device)
            
            encoder_init_hidden = self.encoder.init_hidden(self.device)
            encoder_output, encoder_final_hidden = self.encoder(input_tensor, encoder_init_hidden)
            decoder_input = torch.tensor([[SpecialToken.CLS_token]], device=self.device)
            decoder_hidden = encoder_final_hidden

            print("greedy_search")
            dw, attn, attn_applied = self.greedy_search(decoder_input, decoder_hidden, encoder_output, output_lang)

            if self.hps.heatmap:
                self.plt_heatmap(attn, sentence, dw)
                self.plt_heatmap(attn_applied, sentence, sentence, True)
            return dw, attn, attn_applied

        
    def greedy_search(self, decoder_input, decoder_hidden, encoder_output, output_lang):
        """ Algorithme de recherche"""
        decoded_words = []
        attn_weights_all = []
        attn_applied_all = []
        for di in range(self.hps.max_length):
            if self.hps.decoder_type == 'attention':
                decoder_output, decoder_hidden, attn_weights, attn_applied = self.decoder(decoder_input, decoder_hidden,
                                                                            encoder_output)
                topv, topi = decoder_output.topk(1)
                if topi.item() == SpecialToken.SEP_token:
                    decoded_words.append('SEP')
                    attn_weights_all.append(attn_weights.squeeze().to('cpu').numpy())
                    attn_applied_all.append(attn_applied.squeeze().to('cpu').numpy())
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach() #new futur input is equal to the current word_index predicted
                attn_weights_all.append(attn_weights.squeeze().to('cpu').numpy())
                attn_applied_all.append(attn_applied.squeeze().to('cpu').numpy())
                
            elif self.hps.decoder_type == 'vanilla':
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                topv, topi = decoder_output.topk(1)
                if topi.item() == SpecialToken.SEP_token:
                    decoded_words.append('SEP')
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach() #new futur input is equal to the current word_index predicted
                
            else: 
                raise ValueError('decoder type is illegal')
        return decoded_words, np.array(attn_weights_all), np.array(attn_applied_all)
    

    def plt_heatmap(self, attn, sentence, dw, self_attention=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print("attn.shape:", attn.shape)
        l = len(sentence.split(' ')) + 1
        if self_attention:
            cax = ax.matshow(pd.DataFrame(attn).iloc[:l,:l], cmap='hot')
            fig.colorbar(cax)
            ax.set_yticklabels([''] + sentence.split(' ') + ['SEP'])
        else:
            cax = ax.matshow(pd.DataFrame(attn).iloc[:,:l], cmap='hot')
            fig.colorbar(cax)
            ax.set_yticklabels([''] + dw + ['SEP'])
            
        ax.set_xticklabels([''] + sentence.split(' ') + ['SEP'], rotation=90)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        if self_attention:
            path = os.path.join(self.heat_map_dir, 'attention_{}.png'.format('_'.join([word for word in sentence.split(' ') if word!= '?'])))
        else:
            path = os.path.join(self.heat_map_dir, '{}.png'.format('_'.join([word for word in sentence.split(' ') if word!= '?'])))
        plt.savefig(path)
        return path


    def save_plots(self, train_loss, val_loss):
        """ Function to save the loss and accuracy plots to disk."""
        epoch = np.arange(1, len(train_loss)+1)
        plt.figure(figsize=(10, 7))
        plt.plot(epoch, train_loss, color='blue', linestyle='-',label='train loss')
        plt.plot(epoch, val_loss, color='orange', linestyle='-',label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        path =  os.path.join(os.getcwd(),'photos/outputs_loss.png')
        plt.savefig(path)   
