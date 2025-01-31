import sys
import random
import numpy as np
import torch
import os
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import Datasets
import Transformer
import Encoding


# Function to visualize and print examples from tokens
def decode_example(example, vocab):
    output = ""
    for token in example:
        output += vocab[token] + " "
    return output


# Train Loop
def train(transformer, optimizer, dl_train, epoch, criterion, relative_ids=None, src_vocab_size=None, clip=False, device='cuda'):
    transformer.train()
    print(f"Training epoch: {epoch + 1}")
    epoch_losses = []

    for iter, data_train in enumerate(dl_train):
        source, target = data_train
        source, target = source.to(device), target.to(device)
        target_input = target[:, :-1] # remove last element
        target_real = target[:, 1:] # remove first element
        optimizer.zero_grad()
        # forward pass depending on the model architecture
        if src_vocab_size is None and relative_ids is None:
            predictions = transformer(source, target_input) # Transformer
        elif src_vocab_size is not None and relative_ids is None:
            predictions = torch.log(transformer(source, target_input, src_vocab_size)) # ExtendedStdTransformer2
        elif src_vocab_size is None and relative_ids is not None:
            predictions = transformer(source, target_input, relative_ids[0], relative_ids[1], relative_ids[2]) # ExtendedTransformer1
        else:
            predictions = torch.log(transformer(source, target_input, relative_ids[0], relative_ids[1],
                                                relative_ids[2], src_vocab_size)) # ExtendedTransformer2-4

        predictions_reshape = predictions.contiguous().view(-1, predictions.shape[-1])
        target_real_reshape = target_real.contiguous().view(-1) # (batch_size * target_sequence_length)
        # loss computation
        loss = criterion(predictions_reshape, target_real_reshape) 
        # backward pass
        loss.backward()
        # clip is added to avoid exploding gradient when copy decoder is used
        if clip:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
        # weights and learning rate update
        optimizer.step_and_update_lr()
        epoch_losses.append(loss.detach().cpu().numpy())
        # print iteration loss and mean epoch loss every 500 iterations
        if (iter) % 500 == 0:
            print(f"Iter {iter}, Loss: {loss.item():.6f}")
            print(f"Mean loss: {np.mean(epoch_losses)}")
    print(f"Mean epoch loss: {np.mean(epoch_losses)}")
    
    return np.mean(epoch_losses)

# Accuracy metric, requires predictions as probability distribution
def sequence_level_accuracy(predictions, targets, pad_token=0):
    predicted_tokens = torch.argmax(predictions, dim=-1)
    # Ignore padded tokens in both predictions and targets
    pred_no_pad = predicted_tokens.masked_fill(targets == pad_token, 0)
    targets_no_pad = targets.masked_fill(targets == pad_token, 0) # required if pad_token != 0

    # Compare entire sequences
    match = torch.all(pred_no_pad == targets_no_pad, dim=1)

    # Compute accuracy (proportion of exact matches)
    accuracy = match.float().mean().item()
    return accuracy

# Test loop
def test(transformer, dl_test, relative_ids=None, src_vocab_size=None, device='cuda'):
    transformer.eval()
    sl_accuracies = []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for data_test in dl_test:
            source, target = data_test
            source, target = source.to(device), target.to(device)
            target_input = target[:, :-1]
            target_real = target[:, 1:]
            # forward pass depending on the model architecture
            if src_vocab_size is None and relative_ids is None:
                predictions = F.softmax(transformer(source, target_input), dim=-1) # Transformer
            elif src_vocab_size is not None and relative_ids is None:
                predictions = transformer(source, target_input, src_vocab_size) # ExtendedStdTransformer2
            elif src_vocab_size is None and relative_ids is not None:
                predictions = F.softmax(transformer(source, target_input, relative_ids[0], relative_ids[1],
                                                    relative_ids[2]), dim=-1) # ExtendedTransformer1
            else:
                predictions = transformer(source, target_input, relative_ids[0], relative_ids[1],
                                          relative_ids[2], src_vocab_size) # ExtendedTransformer2-4
            
            # Uncomment to print test loss
            '''
            predictions_reshape = torch.log(predictions).contiguous().view(-1, predictions.shape[-1])
            target_real_reshape = target_real.contiguous().view(-1)
            loss = torch.nn.functional.nll_loss(predictions_reshape, target_real_reshape, ignore_index=0)
            print(f"Test loss: {loss}")
            '''
            # accuracy computation
            accuracy = sequence_level_accuracy(predictions, target_real)
            sl_accuracies.append(accuracy)
    
    print(f"Sequence-level accuracy: {np.mean(sl_accuracies)}")
    return np.mean(sl_accuracies)


# Datasets and Dataloaders
class CustomDataset(Dataset):
    def __init__(self, input_tensor, target_tensor, transform=None, target_transform=None):
        self.input = input_tensor
        self.target = target_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.transform is not None:
            inp = self.transform(self.input[idx])
        else:
            inp = torch.LongTensor(self.input[idx])
        if self.target_transform is not None:
            trg = self.target_transform(self.target[idx])
        else:
            trg = torch.LongTensor(self.target[idx])
        return inp, trg

# Seeds to replicate results
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(111)
seeds = [111, 222, 1111, 2222, 11111]

# Add dataset
print("Add Dataset")
(add_vocab, add_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 add_max_len_inp, add_max_len_trg) = Datasets.create_addition_dataset(200000, 1024)
add_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
add_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
add_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
add_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])
add_dataset_test3 = CustomDataset(input_tensor_val_list[3], target_tensor_val_list[3])

add_train_loader = DataLoader(add_dataset_train, batch_size=64, shuffle=True)
add_test_loader0 = DataLoader(add_dataset_test0, batch_size=64, shuffle=False)
add_test_loader1 = DataLoader(add_dataset_test1, batch_size=64, shuffle=False)
add_test_loader2 = DataLoader(add_dataset_test2, batch_size=64, shuffle=False)
add_test_loader3 = DataLoader(add_dataset_test3, batch_size=64, shuffle=False)

example_train = add_dataset_train.__getitem__(3)
print(f"Add train example from tokens: input {decode_example(example_train[0], add_vocab)}, output {decode_example(example_train[1], add_vocab)}")
print(f"Length of Add Vocabulary: {len(add_vocab)}")

# AddNeg dataset
print("AddNeg Dataset")
(addNeg_vocab, addNeg_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 addNeg_max_len_inp, addNeg_max_len_trg) = Datasets.create_addition_dataset(200000, 1024, negativeProbability=0.25)
addNeg_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
addNeg_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
addNeg_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
addNeg_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])
addNeg_dataset_test3 = CustomDataset(input_tensor_val_list[3], target_tensor_val_list[3])

addNeg_train_loader = DataLoader(addNeg_dataset_train, batch_size=64, shuffle=True)
addNeg_test_loader0 = DataLoader(addNeg_dataset_test0, batch_size=64, shuffle=False)
addNeg_test_loader1 = DataLoader(addNeg_dataset_test1, batch_size=64, shuffle=False)
addNeg_test_loader2 = DataLoader(addNeg_dataset_test2, batch_size=64, shuffle=False)
addNeg_test_loader3 = DataLoader(addNeg_dataset_test3, batch_size=64, shuffle=False)

example_train = addNeg_dataset_train.__getitem__(3)
print(f"AddNeg train example from tokens: input {decode_example(example_train[0], addNeg_vocab)}, output {decode_example(example_train[1], addNeg_vocab)}")
print(f"Length of AddNeg Vocabulary: {len(addNeg_vocab)}")

# Reverse dataset
print("Reverse Dataset")
(reverse_vocab, reverse_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 reverse_max_len_inp, reverse_max_len_trg) = Datasets.create_reversing_dataset(200000, 1024)
reverse_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
reverse_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
reverse_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
reverse_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

reverse_train_loader = DataLoader(reverse_dataset_train, batch_size=64, shuffle=True)
reverse_test_loader0 = DataLoader(reverse_dataset_test0, batch_size=64, shuffle=False)
reverse_test_loader1 = DataLoader(reverse_dataset_test1, batch_size=64, shuffle=False)
reverse_test_loader2 = DataLoader(reverse_dataset_test2, batch_size=64, shuffle=False)

example_train = reverse_dataset_train.__getitem__(1)
print(f"Reverse train example from tokens: input {decode_example(example_train[0], reverse_vocab)}, output {decode_example(example_train[1], reverse_vocab)}")
print(f"Length of Reverse Vocabulary: {len(reverse_vocab)}")

# Dup dataset
print("Dup Dataset")
(dup_vocab, dup_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 dup_max_len_inp, dup_max_len_trg) = Datasets.create_duplicating_dataset(200000, 1024)
dup_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
dup_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
dup_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
dup_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

dup_train_loader = DataLoader(dup_dataset_train, batch_size=64, shuffle=True)
dup_test_loader0 = DataLoader(dup_dataset_test0, batch_size=64, shuffle=False)
dup_test_loader1 = DataLoader(dup_dataset_test1, batch_size=64, shuffle=False)
dup_test_loader2 = DataLoader(dup_dataset_test2, batch_size=64, shuffle=False)

example_train = dup_dataset_train.__getitem__(1)
print(f"Dup train example from tokens: input {decode_example(example_train[0], dup_vocab)}, output {decode_example(example_train[1], dup_vocab)}")
print(f"Length of Dup Vocabulary: {len(dup_vocab)}")

# SCAN-l dataset
print("SCAN-l dataset")
(scanl_vocab, scanl_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 scanl_max_len_inp, scanl_max_len_trg) = Datasets.create_scan_dataset("tasks_train_length.txt", "tasks_test_length.txt")
scanl_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
scanl_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
scanl_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])

scanl_train_loader = DataLoader(scanl_dataset_train, batch_size=64, shuffle=True)
scanl_test_loader0 = DataLoader(scanl_dataset_test0, batch_size=64, shuffle=False)
scanl_test_loader1 = DataLoader(scanl_dataset_test1, batch_size=64, shuffle=False)

example_train = scanl_dataset_train.__getitem__(1)
print(f"SCAN-l train example from tokens: input {decode_example(example_train[0], scanl_vocab)}, output {decode_example(example_train[1], scanl_vocab)}")
print(f"Length of SCAN-l Vocabulary: {len(scanl_vocab)}")

# SCAN-aj dataset
print("SCAN-aj dataset")
(scanaj_vocab, scanaj_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 scanaj_max_len_inp, scanaj_max_len_trg) = Datasets.create_scan_dataset("tasks_train_addprim_jump.txt", "tasks_test_addprim_jump.txt")
scanaj_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
scanaj_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
scanaj_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])

scanaj_train_loader = DataLoader(scanaj_dataset_train, batch_size=64, shuffle=True)
scanaj_test_loader0 = DataLoader(scanaj_dataset_test0, batch_size=64, shuffle=False)
scanaj_test_loader1 = DataLoader(scanaj_dataset_test1, batch_size=64, shuffle=False)

example_train = scanl_dataset_train.__getitem__(1)
print(f"SCAN-aj train example from tokens: input {decode_example(example_train[0], scanaj_vocab)}, output {decode_example(example_train[1], scanaj_vocab)}")
print(f"Length of SCAN-aj Vocabulary: {len(scanaj_vocab)}")

# PCFG-p dataset
print("PCFG-p dataset")
(pcfgp_vocab, pcfgp_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 pcfgp_max_len_inp, pcfgp_max_len_trg) = Datasets.create_pcfg_datset("productivity")
pcfgp_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
pcfgp_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
pcfgp_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])

pcfgp_train_loader = DataLoader(pcfgp_dataset_train, batch_size=64, shuffle=True)
pcfgp_test_loader0 = DataLoader(pcfgp_dataset_test0, batch_size=64, shuffle=False)
pcfgp_test_loader1 = DataLoader(pcfgp_dataset_test1, batch_size=64, shuffle=False)

example_train = pcfgp_dataset_train.__getitem__(1)
print(f"PCFG-p train example from tokens: input {decode_example(example_train[0], pcfgp_vocab)}, output {decode_example(example_train[1], pcfgp_vocab)}")
print(f"Length of PCFG-p Vocabulary: {len(pcfgp_vocab)}")

# PCFG-s dataset
print("PCFG-s dataset")
(pcfgs_vocab, pcfgs_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list,
 pcfgs_max_len_inp, pcfgs_max_len_trg) = Datasets.create_pcfg_datset("systematicity")
pcfgs_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
pcfgs_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
pcfgs_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])

pcfgs_train_loader = DataLoader(pcfgs_dataset_train, batch_size=64, shuffle=True)
pcfgs_test_loader0 = DataLoader(pcfgs_dataset_test0, batch_size=64, shuffle=False)
pcfgs_test_loader1 = DataLoader(pcfgs_dataset_test1, batch_size=64, shuffle=False)

example_train = pcfgs_dataset_train.__getitem__(1)
print(f"PCFG-s train example from tokens: input {decode_example(example_train[0], pcfgs_vocab)}, output {decode_example(example_train[1], pcfgs_vocab)}")
print(f"Length of PCFG-s Vocabulary: {len(pcfgs_vocab)}")


# Learning rate Scheduler
# requires an optimizer, a scale factor for learning rate, the dimension of the model and
# the number of warmup steps (until this number is reached, lr grows linearly, then it decreases
# proportionally to the inverse square root of the step number)
class ScheduledOptim():
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        # Step with the inner optimizer
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # Zero out the gradients with the inner optimizer
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** (-0.5)) * min(n_steps ** (-0.5), n_steps * (n_warmup_steps ** (-1.5)))

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


# Functions to replicate Table1 results
def get_results_Table1(vocab, max_len_enc, max_len_dec, epochs, train_loader,
                       test_loader, test_loader1, test_loader2=None, test_loader3=None, repetitions=1, device='cuda'):
    accuracies_rep = []
    for i in range(repetitions):
        torch.manual_seed(seeds[i])
        accuracies = []
        # models evaluated in Table1
        transformer_abs = Transformer.Transformer(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                  max_seq_length_dec=max_len_dec).to(device)
        transformer_relE = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                            max_seq_length_dec=max_len_dec, attention="rel-e").to(device)
        transformer_relB = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                            max_seq_length_dec=max_len_dec, attention="rel-b").to(device)
        transformer_relEB = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                             max_seq_length_dec=max_len_dec, attention="rel-eb").to(device)
        transformer_rel2E = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                             max_seq_length_dec=max_len_dec, attention="rel2-e").to(device)
        transformer_rel2B = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                             max_seq_length_dec=max_len_dec, attention="rel2-b").to(device)
        transformer_rel2EB = Transformer.ExtendedTransformer1(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        
        transformers_rel = [transformer_relE, transformer_relB, transformer_relEB]
        transformers_rel2 = [transformer_rel2E, transformer_rel2B, transformer_rel2EB]
        # optimizer
        optimizer = ScheduledOptim(torch.optim.Adam(transformer_abs.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
        # Standard Transformer and ExtendedTransformer1 does not have a final Softmax layer in order to use CrossEntropyLoss as criterion
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        # training epochs related to Standard Transformer
        for epoch in range (epochs):
            train_losses = train(transformer_abs, optimizer, train_loader, epoch=epoch, criterion=criterion, device=device)
            # check accuracies at the end of every training epoch
            accuracy0 = test(transformer_abs, test_loader, device=device)
            accuracy1 = test(transformer_abs, test_loader1, device=device)
            accuracy = accuracy1
            if test_loader2 is not None:
                accuracy2 = test(transformer_abs, test_loader2, device=device)
                accuracy = accuracy2
            if test_loader3 is not None:
                accuracy3 = test(transformer_abs, test_loader3, device=device)
                accuracy = accuracy3
        # only the accuracy related to the test set with longer sequences than training ones is saved
        accuracies.append(accuracy)
              
        for transformer in transformers_rel:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            # Standard Transformer and ExtendedTransformer1 does not have a final Softmax layer in order to use CrossEntropyLoss as criterion
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=False)
            # training epochs related to Transformers with rel-* positional encodings
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader,
                                          relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epoch=epoch,
                                          criterion=criterion, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)

        for transformer in transformers_rel2:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            # Standard Transformer and ExtendedTransformer1 does not have a final Softmax layer in order to use CrossEntropyLoss as criterion
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=True)
            # training epochs related to Transformers with rel2-* positional encodings
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader,
                                    relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epoch=epoch,
                                    criterion=criterion, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)
        
        accuracies_rep.append(accuracies)
        
    accuracies_sum = [0] * 7 # 7 is the number of different models trained in Table1 for each dataset
    for i in range(repetitions):
        for j in range(7):
            accuracies_sum[j] += accuracies_rep[i][j]
    # mean accuracy between repetitions for each of the 7 models trained in Table1
    accuracies_mean = [x / repetitions for x in accuracies_sum]

    return accuracies_mean
    
def table1(device):
    # Add dataset
    accuracies_add = get_results_Table1(add_vocab, add_max_len_inp, add_max_len_trg, 2, add_train_loader, add_test_loader0,
                                            add_test_loader1, add_test_loader2, add_test_loader3, repetitions=5, device=device)
    
    # AddNeg dataset
    accuracies_addNeg = get_results_Table1(addNeg_vocab, addNeg_max_len_inp, addNeg_max_len_trg, 10, addNeg_train_loader, addNeg_test_loader0,
                                            addNeg_test_loader1, addNeg_test_loader2, addNeg_test_loader3, repetitions=5, device=device)
    
    # Reverse dataset
    accuracies_reverse = get_results_Table1(reverse_vocab, reverse_max_len_inp, reverse_max_len_trg, 2, reverse_train_loader, reverse_test_loader0,
                                            reverse_test_loader1, reverse_test_loader2, repetitions=5, device=device)
    
    # Dup dataset
    accuracies_dup = get_results_Table1(dup_vocab, dup_max_len_inp, dup_max_len_trg, 4, dup_train_loader, dup_test_loader0,
                                            dup_test_loader1, dup_test_loader2, repetitions=5, device=device)
    
    # SCAN-l dataset
    accuracies_scanl = get_results_Table1(scanl_vocab, scanl_max_len_inp, scanl_max_len_trg, 24, scanl_train_loader, scanl_test_loader0,
                                          scanl_test_loader1, repetitions=3, device=device)
    
    # SCAN-aj dataset
    accuracies_scanaj = get_results_Table1(scanaj_vocab, scanaj_max_len_inp, scanaj_max_len_trg, 24, scanaj_train_loader, scanaj_test_loader0,
                                          scanaj_test_loader1, repetitions=3, device=device)
    
    # PCFG-p dataset
    accuracies_pcfgp = get_results_Table1(pcfgp_vocab, pcfgp_max_len_inp, pcfgp_max_len_trg, 20, pcfgp_train_loader, pcfgp_test_loader0,
                                          pcfgp_test_loader1, repetitions=3, device=device)
    
    # PCFG-s dataset
    accuracies_pcfgs = get_results_Table1(pcfgs_vocab, pcfgs_max_len_inp, pcfgs_max_len_trg, 20, pcfgs_train_loader, pcfgs_test_loader0,
                                          pcfgs_test_loader1, repetitions=3, device=device)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_scanl, accuracies_scanaj, accuracies_pcfgp, accuracies_pcfgs


# Functions to replicate Table 2 results
def get_results_Table2(vocab, max_len_enc, max_len_dec, epochs, train_loader,
                       test_loader, test_loader1, test_loader2=None, test_loader3=None, repetitions=1, device='cuda'):
    accuracies_rep = []
    for i in range(repetitions):
        torch.manual_seed(seeds[i])
        accuracies = []
        # models evaluated in Table2
        transformer_abs_C = Transformer.ExtendedStdTransformer2(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                                max_seq_length_dec=max_len_dec).to(device)
        transformer_relEB_C = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel-eb").to(device)
        transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                                max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        # optimizer
        optimizer = ScheduledOptim(torch.optim.Adam(transformer_abs_C.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
        # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
        # (outputs of the model goes through a torch.log before the loss computation)
        criterion = torch.nn.NLLLoss(ignore_index=0)
        # training epochs related to Standard Transformer with Copy Decoder
        for epoch in range(epochs):
            train_losses = train(transformer_abs_C, optimizer, train_loader, epoch, criterion, src_vocab_size=len(vocab), clip=True, device=device)
            # check accuracies at the end of every training epoch
            accuracy0 = test(transformer_abs_C, test_loader, src_vocab_size=len(vocab), device=device)
            accuracy1 = test(transformer_abs_C, test_loader1, src_vocab_size=len(vocab), device=device)
            accuracy = accuracy1
            if test_loader2 is not None:
                accuracy2 = test(transformer_abs_C, test_loader2, src_vocab_size=len(vocab), device=device)
                accuracy = accuracy2
            if test_loader3 is not None:
                accuracy3 = test(transformer_abs_C, test_loader3, src_vocab_size=len(vocab), device=device)
                accuracy = accuracy3
        # only the accuracy related to the test set with longer sequences than training ones is saved
        accuracies.append(accuracy)
        
        # optimizer
        optimizer = ScheduledOptim(torch.optim.Adam(transformer_relEB_C.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
        # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
        # (outputs of the model goes through a torch.log before the loss computation)
        criterion = torch.nn.NLLLoss(ignore_index=0)
        # relative positional ids
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=False)
        # training epochs related to Transformer with rel-EB positional encodings and Copy Decoder
        for epoch in range(epochs):
            train_losses = train(transformer_relEB_C, optimizer, train_loader, epoch, criterion, 
                                      relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab),
                                      clip=True, device=device)
            # check accuracies at the end of every training epoch
            accuracy0 = test(transformer_relEB_C, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                             src_vocab_size=len(vocab), device=device)
            accuracy1 = test(transformer_relEB_C, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                             src_vocab_size=len(vocab), device=device)
            accuracy = accuracy1
            if test_loader2 is not None:
                accuracy2 = test(transformer_relEB_C, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy2
            if test_loader3 is not None:
                accuracy3 = test(transformer_relEB_C, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy3
        # only the accuracy related to the test set with longer sequences than training ones is saved
        accuracies.append(accuracy)

        # optimizer
        optimizer = ScheduledOptim(torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
        # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
        # (outputs of the model goes through a torch.log before the loss computation)
        criterion = torch.nn.NLLLoss(ignore_index=0)
        # relative positional ids
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=True)
        # training epochs related to Transformer with rel2-EB positional encodings and Copy Decoder
        for epoch in range(epochs):
            train_losses = train(transformer_rel2EB_C, optimizer, train_loader, epoch, criterion,
                                      relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab),
                                      clip=True, device=device)
            # check accuracies at the end of every training epoch
            accuracy0 = test(transformer_rel2EB_C, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                             src_vocab_size=len(vocab), device=device)
            accuracy1 = test(transformer_rel2EB_C, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                             src_vocab_size=len(vocab), device=device)
            accuracy = accuracy1
            if test_loader2 is not None:
                accuracy2 = test(transformer_rel2EB_C, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy2
            if test_loader3 is not None:
                accuracy3 = test(transformer_relEB_C, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy3
        # only the accuracy related to the test set with longer sequences than training ones is saved
        accuracies.append(accuracy)

        accuracies_rep.append(accuracies)
        
    accuracies_sum = [0] * 3 # 3 is the number of different models trained in Table2 for each dataset
    for i in range(repetitions):
        for j in range(3):
            accuracies_sum[j] += accuracies_rep[i][j]
    # mean accuracy between repetitions for each of the 3 models trained in Table2
    accuracies_mean = [x / repetitions for x in accuracies_sum]

    return accuracies_mean

def table2(device):
    
    # Add dataset
    accuracies_add = get_results_Table2(add_vocab, add_max_len_inp, add_max_len_trg, 2, add_train_loader, add_test_loader0,
                                        add_test_loader1, add_test_loader2, add_test_loader3, repetitions=5, device=device)
    
    # AddNeg dataset
    accuracies_addNeg = get_results_Table2(addNeg_vocab, addNeg_max_len_inp, addNeg_max_len_trg, 10, addNeg_train_loader, addNeg_test_loader0,
                                           addNeg_test_loader1, addNeg_test_loader2, addNeg_test_loader3, repetitions=5, device=device)
    
    # Reverse dataset
    accuracies_reverse = get_results_Table2(reverse_vocab, reverse_max_len_inp, reverse_max_len_trg, 2, reverse_train_loader, reverse_test_loader0,
                                            reverse_test_loader1, reverse_test_loader2, repetitions=5, device=device)
    
    # Dup dataset
    accuracies_dup = get_results_Table2(dup_vocab, dup_max_len_inp, dup_max_len_trg, 4, dup_train_loader, dup_test_loader0,
                                        dup_test_loader1, dup_test_loader2, repetitions=5, device=device)
    
    # SCAN-l dataset
    accuracies_scanl = get_results_Table2(scanl_vocab, scanl_max_len_inp, scanl_max_len_trg, 24, scanl_train_loader, scanl_test_loader0,
                                          scanl_test_loader1, repetitions=3, device=device)
    
    # SCAN-aj dataset
    accuracies_scanaj = get_results_Table2(scanaj_vocab, scanaj_max_len_inp, scanaj_max_len_trg, 24, scanaj_train_loader, scanaj_test_loader0,
                                           scanaj_test_loader1, repetitions=3, device=device)
    
    # PCFG-p dataset
    accuracies_pcfgp = get_results_Table2(pcfgp_vocab, pcfgp_max_len_inp, pcfgp_max_len_trg, 20, pcfgp_train_loader, pcfgp_test_loader0,
                                          pcfgp_test_loader1, repetitions=3, device=device)
    
    # PCFG-s dataset
    accuracies_pcfgs = get_results_Table2(pcfgs_vocab, pcfgs_max_len_inp, pcfgs_max_len_trg, 20, pcfgs_train_loader, pcfgs_test_loader0,
                                          pcfgs_test_loader1, repetitions=3, device=device)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_scanl, accuracies_scanaj, accuracies_pcfgp, accuracies_pcfgs


# Functions to replicate Table3 results
def get_results_Table3(vocab, max_len_enc, max_len_dec, epochs, train_loader,
                       test_loader, test_loader1, test_loader2=None, test_loader3=None, repetitions=1, device='cuda'):
    accuracies_rep = []
    for i in range(repetitions):
        torch.manual_seed(seeds[i])
        accuracies = []
        # models evaluated in Table3
        transformer_small4 = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        transformer_small6 = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        transformer_large2 = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        transformer_large4 = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        transformer_large6 = Transformer.ExtendedTransformer2(len(vocab), len(vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=max_len_enc,
                                                              max_seq_length_dec=max_len_dec, attention="rel2-eb").to(device)
        transformers_small = [transformer_small4, transformer_small6]
        transformers_large= [transformer_large2, transformer_large4, transformer_large6]

        for transformer in transformers_small:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
            # (outputs of the model goes through a torch.log before the loss computation)
            criterion = torch.nn.NLLLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=True)
            # training epochs related to Transformer with rel2-EB positional encodings, Copy Decoder, d=64, h=4, f=256 and varying l
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader, epoch, criterion,
                                          relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab),
                                          clip=True, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)
        
        for transformer in transformers_large:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 128, 4000)
            # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
            # (outputs of the model goes through a torch.log before the loss computation)
            criterion = torch.nn.NLLLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                     16, dec2enc_ids=True)
            # training epochs related to Transformer with rel2-EB positional encodings, Copy Decoder, d=128, h=8, f=512 and varying l
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader, epoch, criterion,
                                          relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab), 
                                          clip=True, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)
        
        accuracies_rep.append(accuracies)
    
    accuracies_sum = [0] * 5 # 5 is the number of different models trained in Table3 for each dataset
    for i in range(repetitions):
        for j in range(5):
            accuracies_sum[j] += accuracies_rep[i][j]
    # mean accuracy between repetitions for each of the 5 models trained in Table3
    accuracies_mean = [x / repetitions for x in accuracies_sum]

    return accuracies_mean

def table3(device):
    
    # Add dataset
    accuracies_add = get_results_Table3(add_vocab, add_max_len_inp, add_max_len_trg, 2, add_train_loader, add_test_loader0,
                                        add_test_loader1, add_test_loader2, add_test_loader3, repetitions=5, device=device)
    
    # AddNeg dataset
    accuracies_addNeg = get_results_Table3(addNeg_vocab, addNeg_max_len_inp, addNeg_max_len_trg, 10, addNeg_train_loader, addNeg_test_loader0,
                                           addNeg_test_loader1, addNeg_test_loader2, addNeg_test_loader3, repetitions=5, device=device)
    
    # Reverse dataset
    accuracies_reverse = get_results_Table3(reverse_vocab, reverse_max_len_inp, reverse_max_len_trg, 2, reverse_train_loader, reverse_test_loader0,
                                            reverse_test_loader1, reverse_test_loader2, repetitions=5, device=device)
    
    # Dup dataset
    accuracies_dup = get_results_Table3(dup_vocab, dup_max_len_inp, dup_max_len_trg, 4, dup_train_loader, dup_test_loader0,
                                        dup_test_loader1, dup_test_loader2, repetitions=5, device=device)
    
    # SCAN-l dataset
    accuracies_scanl = get_results_Table3(scanl_vocab, scanl_max_len_inp, scanl_max_len_trg, 24, scanl_train_loader, scanl_test_loader0,
                                          scanl_test_loader1, repetitions=3, device=device)
    
    # SCAN-aj dataset
    accuracies_scanaj = get_results_Table3(scanaj_vocab, scanaj_max_len_inp, scanaj_max_len_trg, 24, scanaj_train_loader, scanaj_test_loader0,
                                           scanaj_test_loader1, repetitions=3, device=device)
    
    # PCFG-p dataset
    accuracies_pcfgp = get_results_Table3(pcfgp_vocab, pcfgp_max_len_inp, pcfgp_max_len_trg, 20, pcfgp_train_loader, pcfgp_test_loader0,
                                          pcfgp_test_loader1, repetitions=3, device=device)
    
    # PCFG-s dataset
    accuracies_pcfgs = get_results_Table3(pcfgs_vocab, pcfgs_max_len_inp, pcfgs_max_len_trg, 20, pcfgs_train_loader, pcfgs_test_loader0,
                                          pcfgs_test_loader1, repetitions=3, device=device)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_scanl, accuracies_scanaj, accuracies_pcfgp, accuracies_pcfgs


# Functions to replicate Table4 results
def get_results_Table4(vocab, max_len_enc, max_len_dec, epochs, train_loader,
                       test_loader, test_loader1, test_loader2=None, test_loader3=None, repetitions=1, device='cuda'):
    accuracies_rep = []
    for i in range(repetitions):
        torch.manual_seed(seeds[i])
        accuracies = []
        # models evaluated in Table4
        transformer_small2s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention= "rel2-eb", shared_weights=True).to(device)
        transformer_small4s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=64, h=4, l=4, f=256, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel2-eb", shared_weights=True).to(device)
        transformer_small6s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=64, h=4, l=6, f=256, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel2-eb", shared_weights=True).to(device)
        transformer_large2s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=128, h=8, l=2, f=512, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel2-eb", shared_weights=True).to(device)
        transformer_large4s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=128, h=8, l=4, f=512, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel2-eb", shared_weights=True).to(device)
        transformer_large6s = Transformer.ExtendedTransformer4(len(vocab), len(vocab), d=128, h=8, l=6, f=512, max_seq_length_enc=max_len_enc,
                                                               max_seq_length_dec=max_len_dec, attention="rel2-eb", shared_weights=True).to(device)
        transformers_small = [transformer_small2s, transformer_small4s, transformer_small6s]
        transformers_large = [transformer_large2s, transformer_large4s, transformer_large6s]

        for transformer in transformers_small:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
            # (outputs of the model goes through a torch.log before the loss computation)
            criterion = torch.nn.NLLLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=True)
            # training epochs related to Transformer with rel2-EB positional encodings, Copy Decoder, d=64, h=4, f=256, varying l and shared weights
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader, epoch, criterion,
                                          relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab),
                                          clip=True, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)
        
        for transformer in transformers_large:
            # optimizer
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9), 1, 128, 4000)
            # since Copy Decoder outputs a probability distribution, Negative Log Likelihood Loss is used as criterion
            # (outputs of the model goes through a torch.log before the loss computation)
            criterion = torch.nn.NLLLoss(ignore_index=0)
            # relative positional ids
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                     16, dec2enc_ids=True)
            # training epochs related to Transformer with rel2-EB positional encodings, Copy Decoder, d=128, h=8, f=512, varying l and shared weights
            for epoch in range(epochs):
                train_losses = train(transformer, optimizer, train_loader, epoch, criterion,
                                          relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(vocab),
                                          clip=True, device=device)
                # check accuracies at the end of every training epoch
                accuracy0 = test(transformer, test_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy1 = test(transformer, test_loader1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                 src_vocab_size=len(vocab), device=device)
                accuracy = accuracy1
                if test_loader2 is not None:
                    accuracy2 = test(transformer, test_loader2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy2
                if test_loader3 is not None:
                    accuracy3 = test(transformer, test_loader3, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids),
                                     src_vocab_size=len(vocab), device=device)
                    accuracy = accuracy3
            # only the accuracy related to the test set with longer sequences than training ones is saved
            accuracies.append(accuracy)
        
        accuracies_rep.append(accuracies)
    
    accuracies_sum = [0] * 6 # 6 is the number of different models trained in Table4 for each dataset
    for i in range(repetitions):
        for j in range(6):
            accuracies_sum[j] += accuracies_rep[i][j]
    # mean accuracy between repetitions for each of the 5 models trained in Table3
    accuracies_mean = [x / repetitions for x in accuracies_sum]

    return accuracies_mean

def table4(device):
    
    # Add dataset
    accuracies_add = get_results_Table4(add_vocab, add_max_len_inp, add_max_len_trg, 2, add_train_loader, add_test_loader0,
                                        add_test_loader1, add_test_loader2, add_test_loader3, repetitions=5, device=device)
    
    # AddNeg dataset
    accuracies_addNeg = get_results_Table4(addNeg_vocab, addNeg_max_len_inp, addNeg_max_len_trg, 10, addNeg_train_loader, addNeg_test_loader0,
                                           addNeg_test_loader1, addNeg_test_loader2, addNeg_test_loader3, repetitions=5, device=device)
    
    # Reverse dataset
    accuracies_reverse = get_results_Table4(reverse_vocab, reverse_max_len_inp, reverse_max_len_trg, 2, reverse_train_loader, reverse_test_loader0,
                                            reverse_test_loader1, reverse_test_loader2, repetitions=5, device=device)
    
    # Dup dataset
    accuracies_dup = get_results_Table4(dup_vocab, dup_max_len_inp, dup_max_len_trg, 4, dup_train_loader, dup_test_loader0,
                                        dup_test_loader1, dup_test_loader2, repetitions=5, device=device)
    
    # SCAN-l dataset
    accuracies_scanl = get_results_Table4(scanl_vocab, scanl_max_len_inp, scanl_max_len_trg, 24, scanl_train_loader, scanl_test_loader0,
                                          scanl_test_loader1, repetitions=3, device=device)
    
    # SCAN-aj dataset
    accuracies_scanaj = get_results_Table4(scanaj_vocab, scanaj_max_len_inp, scanaj_max_len_trg, 24, scanaj_train_loader, scanaj_test_loader0,
                                           scanaj_test_loader1, repetitions=3, device=device)
    
    # PCFG-p dataset
    accuracies_pcfgp = get_results_Table4(pcfgp_vocab, pcfgp_max_len_inp, pcfgp_max_len_trg, 20, pcfgp_train_loader, pcfgp_test_loader0,
                                          pcfgp_test_loader1, repetitions=3, device=device)
    
    # PCFG-s dataset
    accuracies_pcfgs = get_results_Table4(pcfgs_vocab, pcfgs_max_len_inp, pcfgs_max_len_trg, 20, pcfgs_train_loader, pcfgs_test_loader0,
                                          pcfgs_test_loader1, repetitions=3, device=device)

    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_scanl, accuracies_scanaj, accuracies_pcfgp, accuracies_pcfgs


# main
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Table1
    accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1, accuracies_scanl1, accuracies_scanaj1, accuracies_pcfgp1, accuracies_pcfgs1 = table1(device)
    print(f"Table1, Add dataset accuracies: {accuracies_add1}, \nAddNeg dataset accuracies: {accuracies_addNeg1},")
    print(f"Accuracies Reverse dataset: {accuracies_reverse1}, \nAccuracies Dup dataset: {accuracies_dup1},")
    print(f"Accuracis SCAN-l dataset: {accuracies_scanl1}, \nAccuracies SCAN-aj dataset: {accuracies_scanaj1},")
    print(f"Accuracies PCFG-p dataset: {accuracies_pcfgp1}, \nAccuracies PCFG-s dataset: {accuracies_pcfgs1}")
    
    # Table2
    accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_scanl2, accuracies_scanaj2, accuracies_pcfgp2, accuracies_pcfgs2 = table2(device)
    print(f"Table2, Add dataset accuracies: {accuracies_add2}, \nAddNeg dataset accuracies: {accuracies_addNeg2},")
    print(f"Accuracies Reverse dataset: {accuracies_reverse2}, \nAccuracies Dup dataset: {accuracies_dup2},")
    print(f"Accuracis SCAN-l dataset: {accuracies_scanl2}, \nAccuracies SCAN-aj dataset: {accuracies_scanaj2},")
    print(f"Accuracies PCFG-p dataset: {accuracies_pcfgp2}, \nAccuracies PCFG-s dataset: {accuracies_pcfgs2}")
    # Table3
    accuracies_add3, accuracies_addNeg3, accuracies_reverse3, accuracies_dup3, accuracies_scanl3, accuracies_scanaj3, accuracies_pcfgp3, accuracies_pcfgs3 = table3(device)
    print(f"Table3, Add dataset accuracies: {accuracies_add3}, \nAddNeg dataset accuracies: {accuracies_addNeg3},")
    print(f"Accuracies Reverse dataset: {accuracies_reverse3}, \nAccuracies Dup dataset: {accuracies_dup3},")
    print(f"Accuracis SCAN-l dataset: {accuracies_scanl3}, \nAccuracies SCAN-aj dataset: {accuracies_scanaj3},")
    print(f"Accuracies PCFG-p dataset: {accuracies_pcfgp3}, \nAccuracies PCFG-s dataset: {accuracies_pcfgs3}")
    # Table4
    accuracies_add4, accuracies_addNeg4, accuracies_reverse4, accuracies_dup4, accuracies_scanl4, accuracies_scanaj4, accuracies_pcfgp4, accuracies_pcfgs4 = table4(device)
    print(f"Table4, Add dataset accuracies: {accuracies_add4}, \nAddNeg dataset accuracies: {accuracies_addNeg4},")
    print(f"Accuracies Reverse dataset: {accuracies_reverse4}, \nAccuracies Dup dataset: {accuracies_dup4},")
    print(f"Accuracis SCAN-l dataset: {accuracies_scanl4}, \nAccuracies SCAN-aj dataset: {accuracies_scanaj4},")
    print(f"Accuracies PCFG-p dataset: {accuracies_pcfgp4}, \nAccuracies PCFG-s dataset: {accuracies_pcfgs4}")
    
    # Save results in .csv files
    current_dir = os.path.dirname(__file__)
    filepath1 = os.path.join(current_dir, "table1.csv")
    filepath2 = os.path.join(current_dir, "table2.csv")
    filepath3 = os.path.join(current_dir, "table3.csv")
    filepath4 = os.path.join(current_dir, "table4.csv")
    
    res1 = pd.DataFrame(data=[accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1, accuracies_scanl1, accuracies_scanaj1,
                              accuracies_pcfgp1, accuracies_pcfgs1], columns=["abs", "rel-e", "rel-b", "rel-eb", "rel2-e", "rel2-b", "rel2-eb"])
    res1.index = ["Add", "AddNeg", "Reverse", "Dup", "SCAN-l", "SCAN-aj", "PCFG-p", "PCFG-s"]
    res1.to_csv(filepath1)
    
    res2 = pd.DataFrame(data=[accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_scanl2, accuracies_scanaj2,
                              accuracies_pcfgp2, accuracies_pcfgs2], columns=["abs-c", "rel-eb-c", "rel2-eb-c"])
    res2.index = ["Add", "AddNeg", "Reverse", "Dup", "SCAN-l", "SCAN-aj", "PCFG-p", "PCFG-s"]
    res2.to_csv(filepath2)
    
    res3 = pd.DataFrame(data=[accuracies_add3, accuracies_addNeg3, accuracies_reverse3, accuracies_dup3, accuracies_scanl3, accuracies_scanaj3,
                              accuracies_pcfgp3, accuracies_pcfgs3], columns=["small-4", "small-6", "large-2", "large-4", "large-6"])
    res3.index = ["Add", "AddNeg", "Reverse", "Dup", "SCAN-l", "SCAN-aj", "PCFG-p", "PCFG-s"]
    res3.to_csv(filepath3)
    
    res4 = pd.DataFrame(data=[accuracies_add4, accuracies_addNeg4, accuracies_reverse4, accuracies_dup4, accuracies_scanl4, accuracies_scanaj4,
                              accuracies_pcfgp4, accuracies_pcfgs4], columns=["small-2s", "small-4s", "small-6s", "large-2s", "large-4s", "large-6s"])
    res4.index = ["Add", "AddNeg", "Reverse", "Dup", "SCAN-l", "SCAN-aj", "PCFG-p", "PCFG-s"]
    res4.to_csv(filepath4)
    

    sys.exit(0)


if __name__ == "__main__":
    main()