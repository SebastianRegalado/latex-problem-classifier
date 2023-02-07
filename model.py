import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

datapath = "data/training_data.csv"
df = pd.read_csv(datapath)
BATCH_SIZE = 10

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
labels = {0: "algebra", 1: "combinatorics", 2: "geometry", 3: "number theory"}


def tokenize(text: str):
    return tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = [label for label in df["category"]]
        self.texts = [tokenize(text) for text in df["text"]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


# tokenized_text = tokenize("\\begin{itemize} \\item Hola \n \\item $\\sum_{a=0}$ \\end{itemize}")
# print(tokenized_text["input_ids"][0])
# print(tokenizer.decode(tokenized_text["input_ids"][0]))

np.random.seed(112)
df_train, df_val, df_test = np.split(
    df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
)


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    best_val_loss = 0
    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )

        val_loss = total_loss_val / len(val_data)

        if val_loss < best_val_loss or best_val_loss == 0:
            best_val_loss = val_loss
            print(f"Best validation loss: {best_val_loss}")
            print(f"Saving best model for epoch: {epoch_num+1}")
            torch.save(
                {
                    "epoch": epoch_num + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                "outputs/best_model.pth",
            )
        

# EPOCHS = 20
# model = BertClassifier()
# LR = 1e-5

# train(model, df_train, df_val, LR, EPOCHS)

def evaluate(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)
              output = torch.softmax(output, dim=1)
              pred = output.argmax(dim=1)
              for input, label, prediction, out in zip(input_id, test_label, pred, output):
                  print(f"Input: {tokenizer.decode(input)}")
                  print(f"Label: {labels[label.item()]}")
                  print(f"Prediction: {labels[prediction.item()]}")
                  print(f"Prediction score: {out[prediction.item()]}")
                  print()
              acc = (pred == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    
# evaluate(model, df_test)

# Load the best model
checkpoint = torch.load("outputs/best_model.pth")
model = BertClassifier()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
evaluate(model, df_test)