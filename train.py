import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from dataset import CaptchaDataset, get_data
from model import CaptchaNN


def decryption_predictions(preds):
    predictions = []
    last_pred = '*'
    for pred in preds:
        if pred != '*' and last_pred == '*':
            predictions.append(pred)
        last_pred = pred
    return predictions


def train():
    epochs = 100
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 3e-4

    data_train, data_test, chars_set = get_data('data')
    codes, labels = pd.factorize(list(chars_set))
    codes = codes + 1
    chars_dict = dict(zip(labels, codes))
    numb_to_chars = dict(zip(codes, labels))
    numb_to_chars[0] = '*'
    print(f'cd: {chars_dict}')

    model = CaptchaNN(len(chars_set)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(blank=0)

    dataset_train = CaptchaDataset(data_train, chars_dict)
    dataset_test = CaptchaDataset(data_test, chars_dict)

    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        loss_list = []
        for img, labels in loader_train:
            img = img.to(device)
            labels = labels.to(device)
            preds = model(img)
            log_probs = F.log_softmax(preds, dim=2)
            input_length = torch.full(size=(log_probs.shape[1],), fill_value=log_probs.size(0), dtype=torch.int32)
            target_length = torch.full(size=(log_probs.shape[1],), fill_value=labels.size(1), dtype=torch.int32)
            loss = criterion(log_probs, labels, input_length, target_length)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.cpu().detach())
        print(f'loss: {np.mean(loss_list)} epoch: {epoch}')

        with torch.no_grad():
            model.eval()
            true_preds = 0
            count_preds = 0
            for img, labels in loader_test:
                img = img.to(device)
                labels = labels.to(device)

                preds = torch.argmax(model(img), dim=2).cpu().numpy()
                for batch in range(preds.shape[1]):
                    seqs = []
                    for sequence in range(preds.shape[0]):
                        seqs.append(preds[sequence, batch])
                    preds_chars = [numb_to_chars[seq] for seq in seqs]
                    labels_chars = [numb_to_chars[label] for label in labels[batch].cpu().numpy()]
                    preds_decrypted = decryption_predictions(preds_chars)
                    if preds_decrypted == labels_chars:
                        true_preds += 1
                    count_preds += 1
                    print(f'target: {labels_chars} preds: {preds_decrypted}')
            print(f'true: {true_preds} count: {count_preds} accuracy: {true_preds / count_preds}')
            model.train()
