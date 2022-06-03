from dataset import TwitterData
from dataset import Dataset  # for pickle
from model import BertClassifier
from trainer import train, evaluate


def main():
    dataset = TwitterData()
    bert_feed_forward = BertClassifier()
    train(bert_feed_forward, dataset.train_data, dataset.val_data, learning_rate=1e-3, epochs=10, batch_size=64, model_name="BertClassifier")
    evaluate(bert_feed_forward, dataset.test_data, batch_size=64)


if __name__ == "__main__":
    main()
