#!/usr/bin/env python3

import argparse


from math import inf
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models import *
from dataset import get_dataloaders
import neptune
import os


neptune.init(project_qualified_name='j3698/chess')
neptune.create_experiment()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRUNCATION = inf if torch.cuda.is_available() else 20
MAX_EPOCHS = 100
BATCH_SIZE = 1024 if torch.cuda.is_available() else 3
EARLY_STOP = 5 if torch.cuda.is_available() else inf
LR = 0.1 if torch.cuda.is_available() else 1
PROFILE = False


ID = "2"

def main():
    train_dataloader, val_dataloader, _ = get_dataloaders(BATCH_SIZE, 8, truncation = TRUNCATION)
    model = AlphaGoModel2().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr = LR, weight_decay = 5e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, factor = .1, verbose = True)

    best_val_loss = inf
    counter = 0

    neptune.log_text('model', repr(model))

    save_model(model, optimizer, scheduler, inf, 0)
    for i in range(MAX_EPOCHS):
        train_loss, train_correct = train(model, train_dataloader, optimizer, scheduler)
        if PROFILE:
            break

        val_loss, val_correct = evaluate(model, val_dataloader)
        scheduler.step(val_loss if torch.cuda.is_available() else train_loss)

        print(f"EPOCH {i + 1} finished")
        print(f"TRAIN LOSS {train_loss:.3f}, VAL LOSS {val_loss:.3f}")
        print(f"TRAIN CORRECT {train_correct:.3f}, VAL CORRECT {val_correct:.3f}")
        neptune.log_metric('epoch', i + 1)
        neptune.log_metric('train_loss', train_loss)
        neptune.log_metric('val_loss', val_loss)
        neptune.log_metric('train_correct', train_correct)
        neptune.log_metric('val_correct', val_correct)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_model(model, optimizer, scheduler, val_loss, i + 1)
            neptune.log_text('improvement', 'yes')
        else:
            neptune.log_text('improvement', 'no')
            counter += 1
            print("No improvement")
            if counter == EARLY_STOP:
                print("Stopping early")
                neptune.log_text('improvement', 'stopping')
                break

        print()

    print("finished training model")


def save_model(model, optimizer, scheduler, val_loss, epoch):
    path = f"models/{type(model).__name__}-{ID}.pt"
    print(f"Saving model as {path}")
    neptune.log_metric('saved_epochs', epoch + 1)

    if os.path.exists(path):
        raise Exception("Model exists!")

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
    }, path)

def load_model(model_cls, path, optimizer, scheduler):
    checkpoint = torch.load(path)
    model = model_cls()
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    print(f"Epoch: {epoch}")
    print(f"Loss: {loss}")

    return model


def scaled_l2_loss(actual, desired, reduction = 'mean'):
    """
        actual  (batch_size, 1): each value is a predicated value
        desired (batch_size, 2): each row is a value and then uncertainty of that value
    """
    batch, _ = actual.shape
    assert actual.shape == (batch, 1), actual.shape
    assert desired.shape == (batch, 2), desired.shape
    #print("d", desired[:, 0].detach().numpy().tolist())
    #print("a", actual.detach().numpy().tolist())

    std_dev = desired[:, 1, None]
    assert std_dev.shape == (batch, 1), std_dev.shape

    scaled_actual = actual / std_dev
    #print("sa", scaled_actual)
    assert scaled_actual.shape == (batch, 1), scaled_actual.shape

    scaled_desired = desired[:, 0, None] / std_dev
    #print("sd", scaled_desired)
    assert scaled_desired.shape == (batch, 1), scaled_desired.shape

    return F.mse_loss(scaled_actual, scaled_desired, reduction = reduction)


def train(model, train_dataloader, optimizer, scheduler):
    losses_temp = 0
    losses = []
    correct = 0
    total = 0
    model.train()
    for i, (x, y) in enumerate(tqdm.tqdm(train_dataloader)):
        neptune.log_metric('batch_num', i)
        optimizer.zero_grad()

        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)

        loss = scaled_l2_loss(out, y, reduction = 'none')
        #print("l", loss)
        loss_reduced = loss.mean()
        correct += (loss < 1).sum()
        total += len(x)
        losses.append(loss_reduced.item())
        neptune.log_metric('train_loss_1', loss_reduced.item())
        losses_temp += loss_reduced.item()

        loss_reduced.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if i % 60 == 0:
            if i != 0:
                print(f"{losses_temp / 60:.3f}")
                neptune.log_metric('train_loss_60', losses_temp / 60)
            losses_temp = 0


        if i == 20 and PROFILE:
            break

    return np.mean(losses), correct / total


def evaluate(model, eval_dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        losses = []
        for (x, y) in tqdm.tqdm(eval_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = scaled_l2_loss(out, y, reduction = 'none')
            correct += (loss < 1).sum()
            total += len(x)
            losses.append(loss.mean().item())

    return np.mean(losses), correct / total

if __name__ == "__main__":
    main()

