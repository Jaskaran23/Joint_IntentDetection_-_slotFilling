import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pdb
from accuracy import get_accuracy
from data import *
#from model import Encoder, Decoder
from lstm_2layer_model import Encoder, Decoder
import matplotlib.pyplot as plt

#from gru_model import Encoder, Decoder

USE_CUDA = torch.cuda.is_available()


def train(config, train_data, encoder, decoder):
    # loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
    best_decoder = decoder
    best_encoder = encoder
    loss_function_1 = nn.NLLLoss(ignore_index=0)
    loss_function_2 = nn.CrossEntropyLoss()
    enc_optim = optim.Adam(encoder.parameters(), lr=config.learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=config.learning_rate)
    train_loss_over_epochs = []
    val_accuracy_over_epochs_slot = []
    val_accuracy_over_epochs_intent = []
    best_val = 0
    for epoch in range(config.epochs):
        losses = []
        losses_overepoch = []
        count = 0
        for i, batch in enumerate(getBatch(config.batch_size, train_data)):
            count = count+1
            x, y_1, y_2 = zip(*batch)
            x = torch.cat(x)
            tag_target = torch.cat(y_1).view(-1)
            intent_target = torch.cat(y_2)
            x_mask = torch.cat([Variable(torch.BoolTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x])\
                .view(config.batch_size, -1)

            encoder.zero_grad()
            decoder.zero_grad()

            output, hidden_c = encoder(x, x_mask)
            start_decode = Variable(torch.LongTensor([[0]*config.batch_size])).transpose(1, 0)
            # pdb.set_trace()
            tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

            loss_1 = loss_function_1(tag_score, tag_target)
            loss_2 = loss_function_2(intent_score, intent_target)

            #loss = 0.4*loss_1+0.6*loss_2
            loss = loss_1 + loss_2
            losses.append(loss.item())
            losses_overepoch.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 10 == 0:
                print("Epoch", epoch, " batch", i, " : ", np.mean(losses))
                losses = []
        val_accuracy_slot, val_accuracy_intent  = get_accuracy(encoder,decoder)
        print(val_accuracy_slot)
        print(val_accuracy_intent)

        if epoch == 1:
            best_val = val_accuracy_slot
        if val_accuracy_slot > best_val:
            best_val = val_accuracy_slot
            best_decoder = decoder
            best_encoder = encoder
            #best_net = net.parameters
        train_loss_over_epochs.append(np.mean(losses_overepoch))
        val_accuracy_over_epochs_slot.append(val_accuracy_slot)
        val_accuracy_over_epochs_intent.append(val_accuracy_intent)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    
    # pdb.set_trace()
    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.ylabel('Train loss')
    plt.plot(np.arange(config.epochs), train_loss_over_epochs, 'k-')
    plt.title('train loss and slot filling accuracy on validation set')
    plt.xticks(np.arange(config.epochs, dtype=int))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(config.epochs), val_accuracy_over_epochs_slot, 'b-')
    plt.ylabel('Slot filling accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(config.epochs, dtype=int))
    plt.grid(True)
    plt.savefig("plot1.png")

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.ylabel('Train loss')
    plt.plot(np.arange(config.epochs), train_loss_over_epochs, 'k-')
    plt.title('train loss and intent classification accuracy on validation set')
    plt.xticks(np.arange(config.epochs, dtype=int))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(config.epochs), val_accuracy_over_epochs_intent, 'b-')
    plt.ylabel('Intent classification accuracy')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(config.epochs, dtype=int))
    plt.grid(True)
    plt.savefig("plot2.png")

    print('Finished Training')
    torch.save(best_decoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-decoder.pkl'))
    torch.save(best_encoder.state_dict(), os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    print("Train Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.train.w-intent.iob', help='path of train data')
    parser.add_argument('--file_path_val', type=str, default='./data/atis-2.dev.w-intent', help='path of validation data')
    parser.add_argument('--model_dir', type=str, default='./models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of model hidden states')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0007)
    config = parser.parse_args()

    train_data, word2index, tag2index, intent2index = preprocessing(config.file_path, config.max_length)
    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index)//3, config.hidden_size*2)

    encoder.init_weights()
    decoder.init_weights()

    train(config, train_data, encoder, decoder)
