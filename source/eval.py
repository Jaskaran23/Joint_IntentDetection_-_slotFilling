
from sklearn.metrics import f1_score

import argparse

def eval(intent_true1, slot_true1, intent_predict1, slot_predict1):
    predictedintent = []
    predictedslots = []
    trueslots = []
    trueintent = []



    # with open("predictedintent.txt") as file_in:
    #     for line in file_in:
    #         predictedintent.append(int(line))

    # with open("predictedslots.txt") as file_in:
    #     for line in file_in:
    #         predictedslots.append(int(line))

    # with open("trueintent.txt") as file_in:
    #     for line in file_in:
    #         trueintent.append(int(line))

    # with open("trueslots.txt") as file_in:
    #     for line in file_in:
    #         trueslots.append(int(line))



    with open(intent_predict1) as file_in:
        for line in file_in:
            predictedintent.append(int(line))

    with open(slot_predict1) as file_in:
        for line in file_in:
            predictedslots.append(int(line))

    with open(intent_true1) as file_in:
        for line in file_in:
            trueintent.append(int(line))

    with open(slot_true1) as file_in:
        for line in file_in:
            trueslots.append(int(line))
    print("Result on slot filling")
    corrects = 0
    for i in range(len(trueslots)):
        if trueslots[i] == predictedslots[i]:
            corrects = corrects + 1
    acc = corrects/len(trueslots)
    print("accuracy: "+str(acc))
    f1 = f1_score(trueslots, predictedslots, average='weighted')
    print("F1 on slot filling: "+str(f1))
    print("F1 on intent: "+str(f1))
    print("Result on intent classification")
    corrects = 0
    for i in range(len(trueintent)):
        if trueintent[i] == predictedintent[i]:
            corrects = corrects + 1
    acc = corrects/len(trueintent)
    print("accuracy: "+str(acc))
    f1 = f1_score(trueintent, predictedintent, average='weighted')
    print("F1 on intent: "+str(f1))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intent_true', type=str, default='../output/trueintent.txt', help='path of true intent')
    parser.add_argument('--slot_true', type=str, default='../output/trueslots.txt', help='path of true slot')
    parser.add_argument('--intent_predict', type=str, default='../output/predictedintent.txt', help='path of intent predict')

    # Model parameters
    parser.add_argument('--slot_predict', type=str, default='../output/predictedslots.txt', help='path of slot predict')
    config = parser.parse_args()

    intent_true1 = config.intent_true
    slot_true1 = config.slot_true
    intent_predict1 = config.intent_predict
    slot_predict1 = config.slot_predict

    eval(intent_true1, slot_true1, intent_predict1, slot_predict1)


