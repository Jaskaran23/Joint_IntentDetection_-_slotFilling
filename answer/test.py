import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import pdb
from sklearn.metrics import f1_score
from data import *
from gru_model import Encoder, Decoder

def test(processed, some_encoder, some_decoder):
    word2index, tag2index, intent2index = processed

    test_data = get_raw("../data/atis.test.w-intent.iob")

    total_tag = 0
    correct_tag = 0
    total_intent = 0
    correct_intent = 0



    truths = []
    predicteds = []
    trueintents = []
    predictedintents = []
    for index in range(len(test_data)):
    #for index in range(1):
        test_item = test_data[index]
        test_raw, tag_raw, intent_raw = test_item
        # print(intent_raw)
        test_in = prepare_sequence(test_raw,word2index)
        test_mask = Variable(torch.BoolTensor(tuple(map(lambda s: s ==0, test_in)))).view(1,-1)
        start_decode = Variable(torch.LongTensor([[0]*1])).transpose(1,0)

        output, hidden_c = encoder(test_in.unsqueeze(0),test_mask.unsqueeze(0))
        tag_score, intent_score = decoder(start_decode,hidden_c,output,test_mask)

        _, predicted = torch.max(tag_score, dim=1)
        # print(predicted)
        truth = prepare_sequence(tag_raw, tag2index)
        corrects = 0
        for k in range(0, len(truth)):
            if truth[k] == predicted[k]:
                corrects = corrects + 1

        correct_tag += corrects
        total_tag += truth.size(0)

        _, predicted_intent = torch.max(intent_score, dim=1)
        # pdb.set_trace()
        # print(predicted_intent.item())
        intent_raw = intent_raw if intent_raw in intent2index else UNK
        # print(intent_raw)
        true_intent = intent2index[intent_raw]
        # print(true_intent)
        if true_intent == predicted_intent.item():
            correct_intent += 1

        trueintents.append(true_intent.tolist())
        predictedintents.append(predicted_intent.tolist())
        total_intent += 1
        truths.append(truth.tolist())
        predicteds.append(predicted.tolist())

    pred_list = []
    for sublist in predicteds:
        for item in sublist:
            pred_list.append(item)
    truth_list = []
    for sublist in truths:
        for item in sublist:
            truth_list.append(item)

    intent_pred_list = []
    for sublist in predictedintents:
        for item in sublist:
            intent_pred_list.append(item)
    intent_truth_list = []
    for sublist in trueintents:
        #for item in sublist:
        intent_truth_list.append(sublist)
    # print(intent_truth_list)
    # print(intent_pred_list)
    with open("predictedslots.txt", 'w') as output:
        for row in pred_list:
            print(row)
            output.write(str(row))
            output.write('\n')
    # with open("trueslots.txt", 'w') as output:
    #     for row in truth_list:
    #         print(row)
    #         output.write(str(row))
    #         output.write('\n')
    with open("predictedintent.txt", 'w') as output:
        for row in intent_pred_list:
            print(row)
            output.write(str(row))
            output.write('\n')
    # with open("trueintent.txt", 'w') as output:
    #     for row in intent_truth_list:
    #         print(row)
    #         output.write(str(row))
    #         output.write('\n')
    f1 = f1_score(truth_list, pred_list, average='weighted')
    print(f1)
    f1 = f1_score(intent_truth_list, intent_pred_list, average='weighted')
    print(f1)
    print("N =", len(test_data))
    print("Total tag", total_tag, "correct", correct_tag, "accuracy", float(correct_tag/total_tag))
    print("Total intent", total_intent, "correct", correct_intent, "accuracy", float(correct_intent/total_intent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/atis-2.test.w-intent.iob', help='path of train data')
    parser.add_argument('--model_dir', type=str, default='./models/', help='path for saving trained models')

    # Model parameters
    parser.add_argument('--max_length', type=int, default=60, help='max sequence length')
    parser.add_argument('--embedding_size', type=int, default=64, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=64, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    config = parser.parse_args()

    _, word2index,tag2index,intent2index = preprocessing('../data/atis-2.test.w-intent.iob',60)

    encoder = Encoder(len(word2index), config.embedding_size, config.hidden_size)
    decoder = Decoder(len(tag2index), len(intent2index), len(tag2index)//3, config.hidden_size*2)

    encoder.init_weights()
    decoder.init_weights()

    ee = torch.load(os.path.join(config.model_dir, 'jointnlu-encoder.pkl'))
    dd = torch.load(os.path.join(config.model_dir, 'jointnlu-decoder.pkl'))

    encoder.load_state_dict(ee)
    decoder.load_state_dict(dd)
    encoder.eval()
    decoder.eval()
    
    test((word2index, tag2index, intent2index), encoder, decoder)