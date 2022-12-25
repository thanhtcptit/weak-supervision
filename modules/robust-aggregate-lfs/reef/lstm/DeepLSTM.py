import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader  

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from tqdm import tqdm


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class MakeTokens():
	def __init__(self):
		self.max_sentence_length = 500
		self.embedding_vector_length = 32

	def make(self,train_text, val_text, test_text):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(train_text)
		tokenizer.fit_on_texts(val_text)
		tokenizer.fit_on_texts(test_text)
		X_train = tokenizer.texts_to_sequences(train_text)
		X_val = tokenizer.texts_to_sequences(val_text)
		X_test = tokenizer.texts_to_sequences(test_text)
		X_train = sequence.pad_sequences(X_train, maxlen=self.max_sentence_length)
		X_val = sequence.pad_sequences(X_val, maxlen=self.max_sentence_length)
		X_test = sequence.pad_sequences(X_test, maxlen=self.max_sentence_length)
		vocab_size=len(tokenizer.word_index) + 1
		return X_train, X_val, X_test, vocab_size, self.embedding_vector_length, self.max_sentence_length


class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNet, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        # self.linear_3 = nn.Linear(hidden_size, hidden_size)
        # self.linear_4 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.linear_1(x))
        out = F.relu(self.linear_2(out))
        # out = F.relu(self.linear_3(out))
        # out = F.relu(self.linear_4(out))
        out = self.out(out)
        out = self.sig(out)
        # print(out[0:5])
        return out#[:,0]


class DeepLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_vector_length, max_sentence_length, num_classes=1):
        super(DeepLSTM, self).__init__()
        self.hidden_size = 100
        self.embedding_vector_length = embedding_vector_length
        self.max_sentence_length = max_sentence_length
        self.emb = nn.Embedding(vocab_size, embedding_dim = self.embedding_vector_length)
        self.lstm1 = nn.LSTM(input_size = embedding_vector_length, hidden_size = self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        emb = self.emb(x)
        x, _ = self.lstm1(emb)
        x = x[:,-1,:]
        x = self.out(x)
        x = self.sig(x)
        return x

    def get_embedding(self, x):
    	return self.emb(x).view(-1, self.max_sentence_length*self.embedding_vector_length)

def lstm_simple(mode, X_train, y_train, X_test, y_test, vocab_size=768, embedding_vector_length=32,
				max_sentence_length=500, bs=32, epochs=15, num_feats = 500):
	X_train  = torch.tensor(X_train)
	X_test  = torch.tensor(X_test)
	y_train  = torch.tensor(y_train)
	bs = bs
	dataset = TensorDataset(X_train, y_train)
	loader = DataLoader(dataset, batch_size=bs, shuffle=True)
	if mode == "lstm" or mode =='feat_lstm':
		model = DeepLSTM(vocab_size, embedding_vector_length, max_sentence_length).to(device=device) #n_features, n_hidden, n_classes
		supervised_criterion = torch.nn.BCELoss()
	elif mode == "nn":
		model = DeepNet(num_feats, 512, 1).to(device=device)
		supervised_criterion = torch.nn.BCELoss()
	optimizer_lr = torch.optim.Adam(model.parameters(), lr= 0.0003)
	
	epochs = epochs
	
	for i in tqdm(range(epochs)):
	    model.train()
	    for batch_ndx, sample in enumerate(loader):
	    	for i in range(len(sample)):
	    		sample[i] = sample[i].to(device=device)
	    	if mode =='lstm'or mode =='feat_lstm':
	    		loss = supervised_criterion(model(sample[0]), sample[1])
	    	elif mode == 'nn':
	    		loss = supervised_criterion(model(sample[0].float()), sample[1])
	    	loss.backward()
	    	optimizer_lr.step()

	if mode == "lstm" or mode =='feat_lstm':
		probs = model(X_test.to(device=device))
		y_pred = probs.cpu().detach().numpy()
	elif mode =='nn':
		probs = model(X_test.float().to(device=device))
		y_pred = probs.cpu().detach().numpy()

	return y_pred