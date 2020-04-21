from data import *
def get_accuracy(encoder, decoder):

    _, word2index, tag2index, intent2index = preprocessing('./data/atis-2.dev.w-intent', 60)
    test_data = get_raw("./data/atis-2.dev.w-intent.iob")

    total_tag = 0
    correct_tag = 0
    total_intent = 0
    correct_intent = 0



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
        # corrects = 0
        # for k in range(0, len(truth)):
        #     if truth[k] == predicted[k]:
        #         corrects = corrects + 1
        corrects = torch.sum(truth == predicted).item()
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

        total_intent += 1

    slot_tag_accuracy = correct_tag / total_tag
    intent_accuracy = correct_intent / total_intent
    return slot_tag_accuracy, intent_accuracy