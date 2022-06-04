from torch import nn

from dataset import TwitterData
from dataset import Dataset  # for pickle
from model import BertFF, BertAE
from trainer import train, evaluate, geodesic_distance


def train_bert_ff():
    dataset = TwitterData()
    bert_feed_forward = BertFF()
    train(bert_feed_forward,
          dataset.train_data,
          dataset.val_data,
          learning_rate=1e-3,
          criterion=geodesic_distance,
          epochs=10,
          batch_size=64)
    evaluate(bert_feed_forward, dataset.test_data, batch_size=64, criterion=geodesic_distance)


def train_bert_ae():
    dataset = TwitterData()
    bert_autoencoder = BertAE()
    train(bert_autoencoder,
          dataset.train_data,
          dataset.val_data,
          learning_rate=1e-3,
          criterion=nn.MSELoss(),
          epochs=10,
          batch_size=64,
          AE=True)
    evaluate(bert_autoencoder, dataset.test_data, batch_size=64, criterion=geodesic_distance, AE=True)


if __name__ == "__main__":
    train_bert_ae()
