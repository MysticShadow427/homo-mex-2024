# https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/word_att_model.py

import torch
import torch.nn as nn
import sys
import csv
csv.field_size_limit(sys.maxsize)
import torch.nn.functional as F


class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = 100
        self.word_att_net = WordAttNet()
        self.sent_att_net = SentAttNet(sent_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input):

        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)

        return output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        # self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        # self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, f_output,h_output):

        # f_output, h_output = self.gru(input, hidden_state)
        # f_output, h_output Bi-LSTM se aayega
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        # output = self.fc(output)

        return output


class WordAttNet(nn.Module):
    def __init__(self):
        super(WordAttNet, self).__init__()
        # The below code is for initialising word embeddings from scratch to pass to the GRU,we dont need that as we will get direct sequential features from our Bi-LSTM
        # dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        # dict_len, embed_size = dict.shape
        # dict_len += 1
        # unknown_word = np.zeros((1, embed_size))
        # dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        # self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        # self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        # self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        # self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, f_output, h_output):

        # output = self.lookup(input)
        # f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        # f_output and h_output BiLSTM se aayega
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))

        return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return torch.sum(output, 0).unsqueeze(0)