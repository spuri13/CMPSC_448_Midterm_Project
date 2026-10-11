import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_excel("prompt_data.xlsx")


# LLM_output = df['LLM_output']



def run_tests(LLM_text):

    token_list = []

    for output in LLM_text:
        tokens = output.lower().split()
        token_list.append(tokens)

    # print(token_list)

    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    tokens_to_int = []

    for token_ind_list in token_list:
        for token in token_ind_list:
            last_value = next(reversed(word_to_index.values()))
            if token not in word_to_index.keys():
                word_to_index[token] = last_value + 1

    # print(word_to_index)
        
    for token_ind_list in token_list:
        empty_list = []
        for token in token_ind_list:
            empty_list.append(word_to_index[token])
        tokens_to_int.append(empty_list)

    max_length = 0
    for response in tokens_to_int:
        if len(response) > max_length:
            max_length = len(response)

    for int_response in tokens_to_int:
        if len(int_response) != max_length:
            difference = max_length - len(int_response)
            int_response += ([0] * difference) 
    # print(tokens_to_int)


    LLM_names = df["LLM_name"]

    label_to_index = {}

    for LLM_name in LLM_names:
        if LLM_name not in label_to_index.keys():
            label_to_index[LLM_name] = len(label_to_index)

    #print(label_to_index)

    labels_to_int = []

    for LLM_name in LLM_names:
        labels_to_int.append(label_to_index[LLM_name])

    #print(labels_to_int)


    unique_questions = df["LLM_Input"].unique()

    train_questions, test_questions = train_test_split(
        unique_questions,
        test_size=0.2,
        random_state=42
    )

    train_indices = []
    test_indices = []

    for i, question in enumerate(df["LLM_Input"]):
        if question in train_questions:
            train_indices.append(i)
        else:
            test_indices.append(i)

    X_train = []
    for i in train_indices:
        X_train.append(tokens_to_int[i])

    X_test = []
    for i in test_indices:
        X_test.append(tokens_to_int[i])

    Y_train = []
    for i in train_indices:
        Y_train.append(labels_to_int[i])

    Y_test = []
    for i in test_indices:
        Y_test.append(labels_to_int[i])


    X_train = torch.tensor(X_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.long)

    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )


    class CNNClassifier(nn.Module):

        def __init__(self, vocab_size, num_classes):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)

            self.conv = nn.Conv1d(
                in_channels=100,
                out_channels=128,
                kernel_size=3
            )

            self.fc = nn.Linear(128, num_classes)

        def forward(self, x):

            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv(x))
            x = F.max_pool1d(x, kernel_size=x.shape[2])
            x = x.squeeze(2)
            x = self.fc(x)
            return x



    vocab_size = len(word_to_index)
    num_classes = len(label_to_index)

    model = CNNClassifier(vocab_size, num_classes)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )


    epochs = 5

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            train_y_predictions = model(X_batch)
            loss = criterion(train_y_predictions, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()

        print( "Epoch: ", epoch + 1, "Loss:", total_loss / len(train_loader))



    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for X_batch, Y_batch in test_loader:
            test_y_predictions = model(X_batch)

            predicted_classes = torch.argmax(test_y_predictions,dim=1)

            for prediction, actual in zip(predicted_classes, Y_batch):
                if prediction == actual:
                    correct += 1

            total += Y_batch.size(0)


    total_accuracy = correct/total

    print("CNN Accuracy:", total_accuracy)

    print("-----------------------------------------------------------------------------------------------------------------")

    class LSTMClassifier(nn.Module):

        def __init__(self, vocab_size, num_classes):
            super().__init__()

            self.embedding = nn.Embedding(
                vocab_size,
                100,
                padding_idx=0
            )

            self.lstm = nn.LSTM(
                input_size=100,
                hidden_size=128,
                batch_first=True
            )

            self.fc = nn.Linear(128, num_classes)


        def forward(self, x):

            x = self.embedding(x)
            lstm_output, (hidden, cell) = self.lstm(x)
            x = torch.max(lstm_output, dim=1).values
            x = self.fc(x)

            return x


    vocab_size = len(word_to_index)
    num_classes = len(label_to_index)


    model = LSTMClassifier(vocab_size, num_classes)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    epochs = 5
    for epoch in range(epochs):
        model.train()

        total_loss = 0
    
        for X_batch, Y_batch in train_loader:

            optimizer.zero_grad()
            train_y_predictions = model(X_batch)
            loss = criterion(train_y_predictions, Y_batch)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

        print("Epoch: ", epoch + 1, "Loss:", total_loss / len(train_loader))


    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for X_batch, Y_batch in test_loader:

            test_y_predictions = model(X_batch)

            predicted_classes = torch.argmax(test_y_predictions,dim=1)

            for prediction, actual in zip(predicted_classes, Y_batch):
                if prediction == actual:
                    correct += 1

            total += Y_batch.size(0)


    total_accuracy = correct/total

    print("LSTM Accuracy:", total_accuracy)


print("--------------------------------------------------------------------------------------------------")



#Test runs
input_text = df["LLM_Input"]
output_text = df["LLM_output"]
combined_text = df["LLM_Input"] + " " + df["LLM_output"]

print("Input: ")
run_tests(input_text)
print("Output: ")
run_tests(output_text)
print("Input and Output: ")
run_tests(combined_text)



df["response_length"] = df["LLM_output"].apply(lambda x: len(str(x).split()))

def vocabulary_diversity(response):
    words = str(response).lower().split()
    if len(words) == 0:
        return 0
    return len(set(words)) / len(words)

df["vocabulary_diversity"] = df["LLM_output"].apply(vocabulary_diversity)

results = df.groupby("LLM_name")[["response_length", "vocabulary_diversity"]].mean()

print(results)

shortened_output = df["LLM_output"].apply(
    lambda x: " ".join(str(x).split()[:30])
)

print("Shortened Responses:")
run_tests(shortened_output)




