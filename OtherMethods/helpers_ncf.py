import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Change the path to were your data sets are located (train and test)
PATH = 'DATA/'

def load_data():
    test = pd.read_csv(PATH + 'test.csv')
    train = pd.read_csv(PATH + 'train.csv')
    return test, train

def remap_ids(data, user_col, item_col):
    user_mapping = {id_: idx for idx, id_ in enumerate(data[user_col].unique())}
    item_mapping = {id_: idx for idx, id_ in enumerate(data[item_col].unique())}
    
    data[user_col] = data[user_col].map(user_mapping)
    data[item_col] = data[item_col].map(item_mapping)
    
    return data, user_mapping, item_mapping

def scale_ids(train, test):
    user_ids_train = np.array(train['user_id']).reshape(-1, 1)
    book_ids_train = np.array(train['book_id']).reshape(-1, 1)
    ratings_train = np.array(train['rating']).reshape(-1, 1)
    
    user_ids_test = np.array(test['user_id']).reshape(-1, 1)
    book_ids_test = np.array(test['book_id']).reshape(-1, 1)
    to_scale = [user_ids_train, book_ids_train, ratings_train, user_ids_test, book_ids_test]
    scaled = []
    for _, v in enumerate(to_scale):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled.append(scaler.fit_transform(v))
    return scaled[0], scaled[1], scaled[2], scaled[3], scaled[4]

# Create Classes for the train and test set

class TrainSet(Dataset): 
    def __init__(self, user_ids, book_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.book_ids = torch.tensor(book_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'book_id': self.book_ids[idx],
            'rating': self.ratings[idx]
        }

class TestSet(Dataset):
    def __init__(self, user_ids, book_ids):
        self.user_ids = torch.tensor(user_ids, dtype = torch.long)
        self.book_ids = torch.tensor(book_ids, dtype=torch.long)
        
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'book_id': self.book_ids[idx]
        }

# Class for our neural network model
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=8, hidden_dims=[64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        
        # Hidden layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, book):
        user_emb = self.user_embedding(user)
        book_emb = self.book_embedding(book)
        x = torch.cat([user_emb, book_emb], dim=-1)
        x = self.hidden_layers(x)
        return self.sigmoid(self.output_layer(x))