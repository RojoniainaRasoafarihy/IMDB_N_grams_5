# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hkIwtVHvxnVcW3ihFI9zzlgrtrGyPJrD
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import numpy as np
from model import LogisticRegressionModel


class IMDBDataset(Dataset):

    def __init__(self, data, labels):

        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        return self.data[index], self.labels[index]


def train_model(model, dataloader, criterion, optimizer, device):

    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():

    # Charger les données IMDB
    data = load_files("aclImdb/train/", categories=["pos", "neg"], encoding="utf-8", shuffle=True)
    texts, labels = data.data, data.target

    # Prétraitement : TF-IDF avec bi-grams
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(5, 5))
    X = vectorizer.fit_transform(texts).toarray()
    y = np.array(labels)

    # Séparer les données en train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Préparer les datasets et DataLoaders
    train_dataset = IMDBDataset(X_train, y_train)
    val_dataset = IMDBDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialiser le modèle
    input_dim = X.shape[1]
    model = LogisticRegressionModel(input_dim)

    # Configuration du dispositif (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Définir la perte et l'optimiseur
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement et évaluation
    epochs = 5
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")



if __name__ == "__main__":
    main()

