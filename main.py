#Imports
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import Datasets
import Transformer
import Encoding


def loss_function(predictions, target):
    # Create a mask to ignore padding (where real labels are 0)
    mask = (target != 0)

    # Compute the loss using the provided loss function (e.g., cross-entropy)
    loss_ = F.cross_entropy(predictions.permute(0, 2, 1), target, reduction="none") 

    # Apply the mask to ignore padding positions
    mask = mask.float()  # Cast mask to float for compatibility
    loss_ *= mask  # Element-wise multiplication with the mask

    # Calculate the average loss over valid positions
    return loss_.sum() / mask.sum()

# Train and Test functions
def train(transformer, optimizer, scheduler, dl_train, epochs, relative_ids=None, src_vocab_size=None, device='cuda'):
    transformer.train()
    # Save losses of each epoch
    train_losses = []
    for epoch in range(epochs):
        print(f"Training epoch: {epoch}")
        epoch_losses = []
        #for (source, target) in tqdm.tqdm(dl_train, desc=f'Training epoch {epoch}', leave=True):
        for data_train in dl_train:
            source, target = data_train
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()

            if src_vocab_size is None and relative_ids is None:
                predictions = transformer(source, target) # Transformer
            elif src_vocab_size is not None and relative_ids is None:
                predictions = transformer(source, target, src_vocab_size) # ExtendedStdTransformer2
            elif src_vocab_size is None and relative_ids is not None:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2]) # ExtendedTransformer1
            else:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size) # ExtendedTransformer2-4

            #print(f"Predictions: {predictions.shape}, requires_grad: {predictions.requires_grad}")
            #print(f"Target: {target.shape}, requires_grad: {target.requires_grad}")
            #loss = F.cross_entropy(predictions.permute(0, 2, 1), target, reduction="none")
            loss = loss_function(predictions, target)

            #loss.mean().backward()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy())#.item())
        scheduler.step()
        print(f"Mean epoch loss: {np.mean(epoch_losses)}")
        train_losses.append(np.mean(epoch_losses))
    
    return train_losses

# Accuracy metric for testing: an output sequence with a single wrong token is condsidered wrong
def sequence_level_accuracy(predictions, target):
    # Create a mask to ignore padding (where true labels are 0)
    mask = (target != 0)

    # Compute sparse categorical accuracy (get predicted class indices from logits)
    pred_labels = torch.argmax(predictions, dim=-1)
    
    # Compare predicted labels with true labels
    accuracy = (pred_labels == target).float()

    # Apply the mask to ignore padding positions (real == 0)
    accuracy *= mask.float()  # Set accuracy to 0 for positions where mask is False (padding)

    # Return average accuracy over the valid positions
    return (accuracy.sum() / mask.sum())

def test(transformer, dl_test, relative_ids=None, src_vocab_size=None, device='cuda'):
    transformer.eval()
    test_losses = []
    sl_accuracies = []
    print("Testing")
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for data_test in dl_test:
            source, target = data_test
            #print(source.shape)
            #print(target.shape)
            source, target = source.to(device), target.to(device)

            if src_vocab_size is None and relative_ids is None:
                predictions = transformer(source, target) # Transformer
            elif src_vocab_size is not None and relative_ids is None:
                predictions = transformer(source, target, src_vocab_size) # ExtendedStdTransformer2
            elif src_vocab_size is None and relative_ids is not None:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2]) # ExtendedTransformer1
            else:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size) # ExtendedTransformer2-4
                     
            loss_val = F.cross_entropy(predictions.permute(0, 2, 1), target, reduction="none")
            test_losses.append(loss_val.detach().cpu().numpy())
            sl_accuracy = sequence_level_accuracy(predictions, target)
            print(f"Accuracy: {sl_accuracy}")
            sl_accuracies.append(sl_accuracy.detach().cpu())
            
    return np.mean(test_losses), np.mean(sl_accuracies)


# Datasets and Dataloaders
class CustomDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input = input_tensor
        self.target = target_tensor

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]


random.seed(111)
torch.manual_seed(111)
# Add dataset
print("Add Dataset")
(add_vocab, add_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(200000, 1024)
add_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
add_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

add_train_loader = DataLoader(add_dataset_train, batch_size=64, shuffle=True)
add_test_loader = DataLoader(add_dataset_test, batch_size=64, shuffle=False)

# AddNeg dataset
print("AddNeg Dataset")
(addNeg_vocab, addNeg_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(200000, 1024, negativeProbability=0.25)
addNeg_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
addNeg_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

addNeg_train_loader = DataLoader(addNeg_dataset_train, batch_size=64, shuffle=True)
addNeg_test_loader = DataLoader(addNeg_dataset_test, batch_size=64, shuffle=False)

# Reverse dataset
print("Reverse Dataset")
(reverse_vocab, reverse_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_reversing_dataset(200000, 1024)
reverse_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
reverse_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
reverse_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
reverse_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

reverse_train_loader = DataLoader(reverse_dataset_train, batch_size=64, shuffle=True)
reverse_test_loader0 = DataLoader(reverse_dataset_test0, batch_size=64, shuffle=False)
reverse_test_loader1 = DataLoader(reverse_dataset_test1, batch_size=64, shuffle=False)
reverse_test_loader2 = DataLoader(reverse_dataset_test2, batch_size=64, shuffle=False)

# Dup dataset
print("Dup Dataset")
(dup_vocab, dup_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_duplicating_dataset(200000, 1024)
dup_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
dup_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
dup_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
dup_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

dup_train_loader = DataLoader(dup_dataset_train, batch_size=64, shuffle=True)
dup_test_loader0 = DataLoader(dup_dataset_test0, batch_size=64, shuffle=False)
dup_test_loader1 = DataLoader(dup_dataset_test1, batch_size=64, shuffle=False)
dup_test_loader2 = DataLoader(dup_dataset_test2, batch_size=64, shuffle=False)

# Cart dataset
print("Cart Dataset")
(cart_vocab, cart_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_cartesian_dataset(200000, 1024)
cart_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
cart_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
cart_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
cart_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

cart_train_loader = DataLoader(cart_dataset_train, batch_size=64, shuffle=True)
cart_test_loader0 = DataLoader(cart_dataset_test0, batch_size=64, shuffle=False)
cart_test_loader1 = DataLoader(cart_dataset_test1, batch_size=64, shuffle=False)
cart_test_loader2 = DataLoader(cart_dataset_test2, batch_size=64, shuffle=False)

train_source, train_target = next(iter(cart_dataset_train))
cart_max_seq_length_enc = train_source.shape[-1]
cart_max_seq_length_dec = train_target.shape[-1]
test_source1, test_target1 = next(iter(cart_dataset_test1))
cart_max_seq_length_enc1 = test_source1.shape[-1]
cart_max_seq_length_dec1 = test_target1.shape[-1]
test_source2, test_target2 = next(iter(cart_dataset_test2))
cart_max_seq_length_enc2 = test_source2.shape[-1]
cart_max_seq_length_dec2 = test_target2.shape[-1]

# Inters dataset
print("Inters Dataset")
(inters_vocab, inters_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_intersection_dataset(200000, 1024)
inters_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
inters_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
inters_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
inters_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

inters_train_loader = DataLoader(inters_dataset_train, batch_size=64, shuffle=True)
inters_test_loader0 = DataLoader(inters_dataset_test0, batch_size=64, shuffle=False)
inters_test_loader1 = DataLoader(inters_dataset_test1, batch_size=64, shuffle=False)
inters_test_loader2 = DataLoader(inters_dataset_test2, batch_size=64, shuffle=False)

train_source, train_target = next(iter(inters_dataset_train))
inters_max_seq_length_enc = train_source.shape[-1]
inters_max_seq_length_dec = train_target.shape[-1]
test_source1, test_target1 = next(iter(inters_dataset_test1))
inters_max_seq_length_enc1 = test_source1.shape[-1]
inters_max_seq_length_dec1 = test_target1.shape[-1]
test_source2, test_target2 = next(iter(inters_dataset_test2))
inters_max_seq_length_enc2 = test_source2.shape[-1]
inters_max_seq_length_dec2 = test_target2.shape[-1]


# Learning rate scheduler
class CustomSchedule(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model = torch.tensor(self.d_model, dtype=torch.float32)

        super(CustomSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get the current training step
        step = max(1, self.last_epoch)  # Ensure step is at least 1 to avoid division by zero

        # Calculate the learning rate according to the custom schedule
        arg1 = torch.rsqrt(torch.tensor(step, dtype=torch.float32))
        arg2 = step * (self.warmup_steps ** -1.5)

        return [torch.rsqrt(self.d_model) * min(arg1, arg2)]


# Table 1
def table1(device):
    
    # Add
    accuracies_add = []
    print("Table 1 Add dataset")
    transformer_abs = Transformer.Transformer(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, add_train_loader, epochs=2, device=device)
    test_loss, accuracy = test(transformer_abs, add_test_loader, device=device)
    accuracies_add.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_add.append(accuracy)
    

    # AddNeg
    accuracies_addNeg = []
    print("Table 1 AddNeg dataset")
    transformer_abs = Transformer.Transformer(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, addNeg_train_loader, epochs=10, device=device)
    test_loss, accuracy = test(transformer_abs, addNeg_test_loader, device=device)
    accuracies_addNeg.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_addNeg.append(accuracy)

    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_addNeg.append(accuracy)
    

    # Reverse
    accuracies_reverse = []
    print("Table 1 Reverse dataset")
    transformer_abs = Transformer.Transformer(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, reverse_train_loader, epochs=2, device=device)
    test_loss0, accuracy0 = test(transformer_abs, reverse_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, reverse_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, reverse_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2  #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)

    
    # Dup
    accuracies_dup = []
    print("Table 1 Dup dataset")
    transformer_abs = Transformer.Transformer(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, dup_train_loader, epochs=4, device=device)
    test_loss0, accuracy0 = test(transformer_abs, dup_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, dup_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, dup_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)

    
    # Cart
    accuracies_cart = []
    print("Table 1 Cart dataset")
    transformer_abs = Transformer.Transformer(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, cart_train_loader, epochs=4, device=device)
    test_loss0, accuracy0 = test(transformer_abs, cart_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, cart_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, cart_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)

    
    # Inters
    accuracies_inters = []
    print("Table 1 Inters dataset")
    transformer_abs = Transformer.Transformer(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=3).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, inters_train_loader, epochs=8, device=device)
    test_loss0, accuracy0 = test(transformer_abs, inters_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, inters_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, inters_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=True)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters
    

# Table 2
def table2(device):
    # Add
    accuracies_add = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, add_train_loader, epochs=2, src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, add_test_loader, src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)


    # AddNeg
    accuracies_addNeg = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, addNeg_train_loader, epochs=10, src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, addNeg_test_loader, src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)


    # Reverse
    accuracies_reverse = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, reverse_train_loader, epochs=2, src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, reverse_test_loader0, src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, reverse_test_loader1, src_vocab_size=len(reverse_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, reverse_test_loader2, src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)
    

    # Dup
    accuracies_dup = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, dup_train_loader, epochs=4, src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, dup_test_loader0, src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, dup_test_loader1, src_vocab_size=len(dup_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, dup_test_loader2, src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)
    

    # Cart
    accuracies_cart = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, cart_train_loader, epochs=4, src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, cart_test_loader0, src_vocab_size=len(cart_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, cart_test_loader1, src_vocab_size=len(cart_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, cart_test_loader2, src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_relEB_C, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    
    # Inters
    accuracies_inters = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, inters_train_loader, epochs=8, src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, inters_test_loader0, src_vocab_size=len(inters_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, inters_test_loader1, src_vocab_size=len(inters_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, inters_test_loader2, src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_relEB_C, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = accuracy2 #np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters
    

# Table 3
def table3(device):
    # Add
    accuracies_add = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    

    # AddNeg
    accuracies_addNeg = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]
    
    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)


    # Reverse
    accuracies_reverse = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    

    # Dup
    accuracies_dup = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    

    # Cart
    accuracies_cart = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)
    

    # Inters
    accuracies_inters = []
    transformer_small4 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small6 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large2 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large4 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large6 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters


# Table 4
def table4(device):
    # Add
    accuracies_add = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    

    # AddNeg
    accuracies_addNeg = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]
    
    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)


    # Reverse
    accuracies_reverse = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    

    # Dup
    accuracies_dup = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    

    # Cart
    accuracies_cart = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)
    

    # Inters
    accuracies_inters = []
    transformer_small2s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small4s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_small6s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large2s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large4s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformer_large6s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1, accuracies_cart1, accuracies_inters1 = table1(device)
    accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_cart2, accuracies_inters2 = table2(device)
    #accuracies_add3, accuracies_addNeg3, accuracies_reverse3, accuracies_dup3, accuracies_cart3, accuracies_inters3 = table3(device)
    #accuracies_add4, accuracies_addNeg4, accuracies_reverse4, accuracies_dup4, accuracies_cart4, accuracies_inters4 = table4(device)
    print(f"Table 1, Add dataset accuracies: {accuracies_add1},\nAddNeg dataset accuracies: {accuracies_addNeg1},\nReverse dataset accuracies: {accuracies_reverse1},\nDup dataset accuracies: {accuracies_dup1},\nCart dataset accuracies: {accuracies_cart1},\nInters dataset accuracies: {accuracies_inters1}")
    print(f"Table 2, Add dataset accuracies: {accuracies_add2},\nAddNeg dataset accuracies: {accuracies_addNeg2},\nReverse dataset accuracies: {accuracies_reverse2},\nDup dataset accuracies: {accuracies_dup2},\nCart dataset accuracies: {accuracies_cart2},\nInters dataset accuracies: {accuracies_inters2}")
    res1 = pd.DataFrame(data=[accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1, accuracies_cart1, accuracies_inters1], columns=["abs", "rel-e", "rel-b", "rel-eb", "rel2-e", "rel2-b", "rel2-eb"])
    res1.index = ["Add", "AddNeg", "Reverse", "Dup", "Cart", "Inters"]
    res1.to_csv(".../table1.csv")
    res2 = pd.DataFrame(data=[accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_cart2, accuracies_inters2], columns=["rel-eb", "rel2-eb", "abs-c", "rel-eb-c", "rel2-eb-c"])
    res2.index = ["Add", "AddNeg", "Reverse", "Dup", "Cart", "Inters"]
    res2.to_csv(".../table2.csv")
    #print(f"Table 3, Add dataset accuracies: {accuracies_add3},\nAddNeg dataset accuracies: {accuracies_addNeg3},\nReverse dataset accuracies: {accuracies_reverse3},\nDup dataset accuracies: {accuracies_dup3},\nCart dataset accuracies: {accuracies_cart3},\nInters dataset accuracies: {accuracies_inters3}")
    #print(f"Table 4, Add dataset accuracies: {accuracies_add4},\nAddNeg dataset accuracies: {accuracies_addNeg4},\nReverse dataset accuracies: {accuracies_reverse4},\nDup dataset accuracies: {accuracies_dup4},\nCart dataset accuracies: {accuracies_cart4},\nInters dataset accuracies: {accuracies_inters4}")
    

    # Try transformer
    #transformer = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=512, h=8, l=6, f=2048, max_seq_length_enc=25, max_seq_length_dec=50, dropout=0.0).to('cpu')
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    #scheduler = CustomSchedule(optimizer, d_model=512, warmup_steps=4000)
    #enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, False)
    #train_losses = train(transformer, optimizer, scheduler, dup_train_loader, 2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #torch.save(transformer.state_dict(), './transformer_try.pth')
    #print(train_losses)
    #transformer.load_state_dict(torch.load('./transformer_try.pth', map_location=torch.device('cpu')))

    #loss0, accuracy0 = test(transformer, dup_test_loader0, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #loss1, accuracy1 = test(transformer, dup_test_loader1, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, False)
    #loss2, accuracy2 = test(transformer, dup_test_loader2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #loss = np.mean([loss0, loss1, loss2])
    #accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    #print(f"Average loss: {loss}, average accuracy: {accuracy}")
    
    # Try transformer1
    #transformer = Transformer.ExtendedTransformer1(len(add_vocab), len(add_vocab), d=512, h=8, l=6, f=2048, max_seq_length_enc=26, max_seq_length_dec=14, dropout=0.0).to('cpu')
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    #scheduler = CustomSchedule(optimizer, d_model=512, warmup_steps=4000)
    #enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, False)
    #train_losses = train(transformer, optimizer, scheduler, add_train_loader, 2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #torch.save(transformer.state_dict(), './transformer_try.pth')
    #print(train_losses)
    #transformer.load_state_dict(torch.load('./transformer_try.pth', map_location=torch.device('cpu')))

    #loss, accuracy = test(transformer, add_test_loader, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    #print(f"Average loss: {loss}, average accuracy: {accuracy}")
    
    # Try transformer2
    #transformer = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=512, h=8, l=6, f=2048, max_seq_length_enc=26, max_seq_length_dec=14, dropout=0.0)
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    #scheduler = CustomSchedule(optimizer, d_model=512, warmup_steps=4000)
    #enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, False)
    #train_losses = train(transformer, optimizer, scheduler, add_train_loader, 2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), len(add_vocab))
    #torch.save(transformer.state_dict(), './transformer_try.pth')
    #print(train_losses)
    #transformer.load_state_dict(torch.load('./transformer_try.pth', map_location=torch.device('cpu')))

    #loss, accuracy = test(transformer, add_test_loader, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), len(add_vocab))
    #print(f"Average loss: {loss}, average accuracy: {accuracy}")
    
    # Try transformer4
    #transformer = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=512, h=8, l=6, f=2048, max_seq_length_enc=26, max_seq_length_dec=14, dropout=0.0, shared_weights=True)
    #optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    #scheduler = CustomSchedule(optimizer, d_model=512, warmup_steps=4000)
    #enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, False)
    #train_losses = train(transformer, optimizer, scheduler, add_train_loader, 2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), len(add_vocab))
    #torch.save(transformer.state_dict(), './transformer_try.pth')
    #print(train_losses)
    #transformer.load_state_dict(torch.load('./transformer_try.pth', map_location=torch.device('cpu')))

    #loss, accuracy = test(transformer, add_test_loader, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), len(add_vocab))
    #print(f"Average loss: {loss}, average accuracy: {accuracy}")
    #for i in range(3):
     #   train_input, train_targets = next(iter(add_dataset_train))
      #  enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, False)
       # output = model(train_input, train_targets, enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids, len(add_vocab))
        #print(output)

    sys.exit(0)

if __name__ == "__main__":
    main()