#Imports
import sys
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import Datasets
import Transformer
import Encoding


# Train and Test functions
def train(transformer, optimizer, scheduler, dl_train, epochs, relative_ids=None, src_vocab_size=None, device='cpu'):
    transformer.train()
    # Save losses of each epoch
    train_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for (source, target) in tqdm.tqdm(dl_train, desc=f'Training epoch {epoch}', leave=True):
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

            print(f"Predictions: {predictions.shape}, requires_grad: {predictions.requires_grad}")
            print(f"Target: {target.shape}, requires_grad: {target.requires_grad}")
            loss = F.cross_entropy(predictions.permute(0, 2, 1), target, reduction="none")

            loss.mean().backward()
            optimizer.step()
            epoch_losses.append(loss.detach().numpy())#.item())
        scheduler.step()
        train_losses.append(np.mean(epoch_losses))
    
    return train_losses

# Accuracy metric for testing: an output sequence with a single wrong token is condsidered wrong
def sequence_level_accuracy(predictions, target):
    # Get the predicted indexes by taking the argmax along the last dimension (tokens)
    pred_tokens = torch.argmax(predictions, dim=-1)
    
    # Compare the predicted tokens with the target and compute the mean accuracy
    accuracy = (pred_tokens == target).float().mean()
    
    return accuracy

def test(transformer, dl_test, relative_ids=None, src_vocab_size=None, device='cpu'):
    transformer.eval()
    test_losses = []
    sl_accuracies = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for data_test in dl_test:
            source, target = data_test
            print(source.shape)
            print(target.shape)
            #y_val_conf.extend(y_val.detach().cpu().numpy()) # Collect ground truths
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
            test_losses.append(loss_val.detach().numpy())
            sl_accuracy = sequence_level_accuracy(predictions, target)
            sl_accuracies.append(sl_accuracy)
            
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

# Add dataset
print("Add Dataset")
(add_vocab, add_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(192, 128)
add_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
add_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

add_train_loader = DataLoader(add_dataset_train, batch_size=64, shuffle=True)
add_test_loader = DataLoader(add_dataset_test, batch_size=64, shuffle=False)

# AddNeg dataset
print("AddNeg Dataset")
(addNeg_vocab, addNeg_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(192, 128, negativeProbability=0.25)
addNeg_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
addNeg_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

addNeg_train_loader = DataLoader(addNeg_dataset_train, batch_size=64, shuffle=True)
addNeg_test_loader = DataLoader(addNeg_dataset_test, batch_size=64, shuffle=False)

# Reverse dataset
print("Reverse Dataset")
(reverse_vocab, reverse_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_reversing_dataset(192, 128)
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
(dup_vocab, dup_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_duplicating_dataset(192, 128)
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
(cart_vocab, cart_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_cartesian_dataset(192, 128)
cart_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
cart_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
cart_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
cart_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

cart_train_loader = DataLoader(cart_dataset_train, batch_size=64, shuffle=True)
cart_test_loader0 = DataLoader(cart_dataset_test0, batch_size=64, shuffle=False)
cart_test_loader1 = DataLoader(cart_dataset_test1, batch_size=64, shuffle=False)
cart_test_loader2 = DataLoader(cart_dataset_test2, batch_size=64, shuffle=False)

# Inters dataset
print("Inters Dataset")
(inters_vocab, inters_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_intersection_dataset(192, 128)
inters_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
inters_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
inters_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
inters_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

inters_train_loader = DataLoader(inters_dataset_train, batch_size=64, shuffle=True)
inters_test_loader0 = DataLoader(inters_dataset_test0, batch_size=64, shuffle=False)
inters_test_loader1 = DataLoader(inters_dataset_test1, batch_size=64, shuffle=False)
inters_test_loader2 = DataLoader(inters_dataset_test2, batch_size=64, shuffle=False)


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


# Table1
def table1(device):
    # Add
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

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
    

    # AddNeg
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

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)

    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)


    # Reverse
    transformer_abs = Transformer.Transformer(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    tranformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, reverse_train_loader, epochs=2, device=device)
    test_loss0, accuracy0 = test(transformer_abs, reverse_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, reverse_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, reverse_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

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
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
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
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


    # Dup
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
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

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
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
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
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


    # Cart
    transformer_abs = Transformer.Transformer(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, cart_train_loader, epochs=4, device=device)
    test_loss0, accuracy0 = test(transformer_abs, cart_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, cart_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, cart_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


    # Inters
    transformer_abs = Transformer.Transformer(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3).to(device)
    transformer_relE = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel-e").to(device)
    transformer_relB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel-b").to(device)
    transformer_relEB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel-eb").to(device)
    transformer_rel2E = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-e").to(device)
    transformer_rel2B = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-b").to(device)
    transformer_rel2EB = Transformer.ExtendedTransformer1(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb").to(device)
    
    transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
    transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, inters_train_loader, epochs=8, device=device)
    test_loss0, accuracy0 = test(transformer_abs, inters_test_loader0, device=device)
    test_loss1, accuracy1 = test(transformer_abs, inters_test_loader1, device=device)
    test_loss2, accuracy2 = test(transformer_abs, inters_test_loader2, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


def table2(device):
    # Add
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, add_train_loader, epochs=2, src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, add_test_loader, src_vocab_size=len(add_vocab), device=device)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)


    # AddNeg
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, addNeg_train_loader, epochs=10, src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, addNeg_test_loader, src_vocab_size=len(addNeg_vocab), device=device)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)


    # Reverse
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, reverse_train_loader, epochs=2, src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, reverse_test_loader0, src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, reverse_test_loader1, src_vocab_size=len(reverse_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, reverse_test_loader2, src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, reverse_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, reverse_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, reverse_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    

    # Dup
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, dup_train_loader, epochs=4, src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, dup_test_loader0, src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, dup_test_loader1, src_vocab_size=len(dup_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, dup_test_loader2, src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, dup_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, dup_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, dup_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])


    # Cart
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, cart_train_loader, epochs=4, src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, cart_test_loader0, src_vocab_size=len(cart_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, cart_test_loader1, src_vocab_size=len(cart_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, cart_test_loader2, src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])


    # Inters
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel-eb")
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, inters_train_loader, epochs=8, src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, inters_test_loader0, src_vocab_size=len(inters_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, inters_test_loader1, src_vocab_size=len(inters_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, inters_test_loader2, src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])


def table3(device):
    # Add
    transformer_small4 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    

    # AddNeg
    transformer_small4 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]
    
    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)


    # Reverse
    transformer_small4 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
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
    

    # Dup
    transformer_small4 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
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
    

    # Cart
    transformer_small4 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    

    # Inters
    transformer_small4 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_small6 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large2 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large4 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large6 = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformers_small = [transformer_small4, transformer_small6]
    transformers_large = [transformer_large2, transformer_large4, transformer_large6]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


def table4(device):
    # Add
    transformer_small2s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(add_vocab), len(add_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    

    # AddNeg
    transformer_small2s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(addNeg_vocab), len(addNeg_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb")
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]
    
    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)


    # Reverse
    transformer_small2s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(reverse_vocab), len(reverse_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb")
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
    

    # Dup
    transformer_small2s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, g=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(dup_vocab), len(dup_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb")
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
    

    # Cart
    transformer_small2s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(cart_vocab), len(cart_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=18, max_seq_length_dec=194, attention="rel2-eb")
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(14, 110, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(18, 194, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    

    # Inters
    transformer_small2s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_small4s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_small6s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large2s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large4s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformer_large6s = Transformer.ExtendedTransformer4(len(inters_vocab), len(inters_vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=48, max_seq_length_dec=3, attention="rel2-eb")
    transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
    transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

    for transformer in transformers_small:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomSchedule(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(33, 3, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(48, 3, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Try transformer
    transformer = Transformer.ExtendedTransformer1(len(dup_vocab), len(dup_vocab), d=512, h=8, l=6, f=2048, max_seq_length_enc=25, max_seq_length_dec=50, dropout=0.0).to('cpu')
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomSchedule(optimizer, d_model=512, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, False)
    train_losses = train(transformer, optimizer, scheduler, dup_train_loader, 2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    torch.save(transformer.state_dict(), './transformer_try.pth')
    print(train_losses)
    transformer.load_state_dict(torch.load('./transformer_try.pth', map_location=torch.device('cpu')))

    loss0, accuracy0 = test(transformer, dup_test_loader0, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    loss1, accuracy1 = test(transformer, dup_test_loader1, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, False)
    loss2, accuracy2 = test(transformer, dup_test_loader2, (enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids))
    loss = np.mean([loss0, loss1, loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])

    print(f"Average loss: {loss}, average accuracy: {accuracy}")
    
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