#Imports
import sys
import random
import numpy as np
import torch
import os
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import Datasets
import Transformer
import Encoding


# Function to visualize examples better from tokens
def decode_example(example, vocab):
    output = ""
    for token in example:
        output += vocab[token] + " "
    return output


# Train and Test functions
def train_step(transformer, optimizer, dl_train, epoch, criterion, relative_ids=None, src_vocab_size=None, clip=False, device='cuda'):
    transformer.train()
    # Save losses of each epoch
    train_losses = []
    #for epoch in range(epochs):
    print(f"Training epoch: {epoch}")
    epoch_losses = []
    #for (source, target) in tqdm.tqdm(dl_train, desc=f'Training epoch {epoch}', leave=True):
    for iter, data_train in enumerate(dl_train):
        source, target = data_train
        source, target = source.to(device), target.to(device)
        target_input = target[:, :-1] # remove last element
        target_real = target[:, 1:] # remove first element
        optimizer.zero_grad()

        if src_vocab_size is None and relative_ids is None:
            predictions = transformer(source, target_input) # Transformer
        elif src_vocab_size is not None and relative_ids is None:
            predictions = transformer(source, target_input, src_vocab_size) # ExtendedStdTransformer2
        elif src_vocab_size is None and relative_ids is not None:
            predictions = transformer(source, target_input, relative_ids[0], relative_ids[1], relative_ids[2]) # ExtendedTransformer1
        else:
            predictions = transformer(source, target_input, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size) # ExtendedTransformer2-4

        #print(f"Predictions: {predictions.shape}, requires_grad: {predictions.requires_grad}")
        #print(f"Target: {target.shape}, requires_grad: {target.requires_grad}")
        #loss = F.cross_entropy(predictions, target, ignore_index=0, reduction="mean")
        predictions = predictions.view(-1, predictions.size(-1)) # (batch_size * target_sequence_length, vocab_size)
        #target_real = target_real.reshape(-1) # (batch_size * target_sequence_length)
        target_real = target_real.contiguous().view(-1)
        loss = criterion(predictions, target_real) 

        loss.backward()
        #optimizer.step()
        #scheduler.step()
        if clip:
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1)
        optimizer.step_and_update_lr()
        epoch_losses.append(loss.detach().cpu().numpy())#.item())
        if (iter) % 500 == 0:
            print(f"Loss of iter {iter}: {loss.item():.3f}")
    print(f"Mean epoch loss: {np.mean(epoch_losses)}")
    train_losses.append(np.mean(epoch_losses))
    
    return train_losses


def sequence_level_accuracy(predictions, target, eos_token_idx=2, greedy=False):
    if not greedy:
        predicted_token = torch.argmax(predictions, dim=-1) # (batch_size, seq_len)
    else:
        predicted_token = predictions
    target_list = target.tolist()
    pred_token_list = predicted_token.tolist()

    total_correct = 0
    for i in range(len(target_list)): # batch
        correct = 1
        for j in range(len(target_list[i])): # element of batch
            if target_list[i][j] != eos_token_idx:
                if target_list[i][j] != pred_token_list[i][j]:
                    correct = 0
                    break
            elif target_list[i][j] == pred_token_list[i][j]:
                break
            else:
                correct = 0
                break
        total_correct += correct
    
    accuracy = total_correct / len(target_list)
    return accuracy


def test(transformer, dl_test, max_trg_len, max_src_len=0, relative_ids=None, src_vocab_size=None, device='cuda'):
    transformer.eval()
    sl_accuracies = []
    sl_accuracies_gd = []
    print("Testing")
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for data_test in dl_test:
            source, target = data_test
            #print(source.shape)
            #print(target.shape)
            source, target = source.to(device), target.to(device)
            target_input = target[:, :-1]
            target_real = target[:, 1:]

            if src_vocab_size is None and relative_ids is None:
                predictions = transformer(source, target_input, training=False) # Transformer
                predictions_gd = greedy_decode(transformer, source, max_trg_len, max_src_len, 3, 2)
            elif src_vocab_size is not None and relative_ids is None:
                predictions = transformer(source, target_input, src_vocab_size, training=False) # ExtendedStdTransformer2
                predictions_gd = greedy_decode(transformer, source, max_trg_len, max_src_len, 3, 2, src_vocab_size=src_vocab_size)
            elif src_vocab_size is None and relative_ids is not None:
                predictions = transformer(source, target_input, relative_ids[0], relative_ids[1], relative_ids[2], training=False) # ExtendedTransformer1
                predictions_gd = greedy_decode(transformer, source, max_trg_len, max_src_len, 3, 2, relative_ids=relative_ids)
            else:
                predictions = transformer(source, target_input, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size, training=False) # ExtendedTransformer2-4
                predictions_gd = greedy_decode(transformer, source, max_trg_len, max_src_len, 3, 2, relative_ids=relative_ids, src_vocab_size=src_vocab_size)
            
            accuracy = sequence_level_accuracy(predictions, target_real)
            sl_accuracies.append(accuracy)
            accuracy_gd = sequence_level_accuracy(predictions_gd, target_real, greedy=True)
            sl_accuracies_gd.append(accuracy_gd)
    
    print(f"Sequence-level accuracy: {np.mean(sl_accuracies)}")
    print(f"Sequence-level accuracy with Greedy Decoding: {np.mean(sl_accuracies_gd)}")
    return np.mean(sl_accuracies)

def greedy_decode(transformer, src, max_len_trg, max_len_src, start_symbol, eos_token, relative_ids=None, src_vocab_size=None, device='cuda'):
    transformer.eval()
    batch_size = src.shape[0]
    trg = torch.zeros(batch_size, max_len_trg).fill_(start_symbol).type(torch.long).to(device)

    with torch.no_grad():
        src_embedded = transformer.encoder_embedding(src)
        src_embedded = transformer.positional_encoding_enc(src_embedded)
        src_mask = transformer.create_padding_mask(src)
        if relative_ids is not None:
            enc_output = transformer.encoder(src_embedded, relative_ids[0], src_mask, training=False)
        else:
            enc_output = transformer.encoder(src_embedded, src_mask, training=False)
    
    for i in range(max_len_trg - 1):
        target = trg[:, :i + 1]

        with torch.no_grad():
            trg_embedded = transformer.decoder_embedding(target)
            trg_embedded = transformer.positional_encoding_dec(trg_embedded)
            src_mask = transformer.create_padding_mask(src)
            look_ahead_mask = transformer.create_look_ahead_mask(target.shape[1])
            dec_trg_padding_mask = transformer.create_padding_mask(target)
            trg_mask = torch.max(look_ahead_mask, dec_trg_padding_mask)
            if relative_ids is not None:
                relative_ids = Encoding.create_relative_ids(max_len_src, i + 2, 16, False)
                dec_output = transformer.decoder(trg_embedded, enc_output, relative_ids[1], relative_ids[2], trg_mask, src_mask, training=False)
            else:
                dec_output = transformer.decoder(trg_embedded, enc_output, trg_mask, src_mask, training=False)
            output = F.softmax(transformer.fc(dec_output), dim=-1)
        
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
        trg[:, i + 1] = next_token.squeeze()
        
        if (next_token == eos_token).all():
            break
    print(trg[0])
    return trg
    

'''
def sequence_accuracy(predictions, targets, pad_token):
    # Ensure predictions and targets are both tensors for comparison
    #if isinstance(predictions, list):
    #    predictions = torch.stack(predictions)  # Convert list of tensors to a single tensor if needed

    # Strip out padding tokens from both predictions and targets
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Initialize counters
    correct_sequences = 0
    total_sequences = predictions.shape[0]  # Number of sequences

    # Loop through each sequence and compare
    for i in range(total_sequences):
        # Remove padding tokens for both prediction and target
        pred_seq = [token for token in predictions[i] if token != pad_token]
        trg_seq = [token for token in targets[i] if token != pad_token]
        # Check if both sequences match exactly
        if pred_seq == trg_seq:
            correct_sequences += 1

    # Calculate the accuracy
    accuracy = correct_sequences / total_sequences

    return accuracy * 100

def token_accuracy(predictions, targets, pad_token):
    # Ensure predictions and targets are on CPU for comparison
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Mask padding tokens in targets
    mask = targets != pad_token       # True for non-pad tokens

    # Count correct token predictions (ignoring padding)
    correct_tokens = (predictions == targets) & mask
    total_valid_tokens = mask.sum()  # Total valid (non-padding) tokens

    # Calculate accuracy as a percentage
    accuracy = correct_tokens.sum() / total_valid_tokens * 100 if total_valid_tokens > 0 else 0

    return accuracy

# Testing function
def test_transformer_with_accuracy(model, dataloader, pad_token, start_symbol, max_len_trg, max_len_src=0, relative_ids=None, src_vocab_size=None, device='cuda', add=False):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data_test in dataloader:
            src, trg = data_test
            src, trg = src.to(device), trg.to(device)
            target_input = trg[:, :-1]
            target_real = trg[:, 1:]

            # Greedy decode to generate predictions
            predicted_trg = greedy_decode(model, src, max_len_trg, max_len_src, start_symbol, pad_token, relative_ids, src_vocab_size, device)
            #print(f"Prediction: {predicted_trg.tolist()[0]}, Target: {target_real.tolist()[0]}")
            # Store the predicted sequences and targets
            all_predictions.append(predicted_trg[:, 1:])
            all_targets.append(target_real)

    # Flatten predictions and targets to a single tensor
    all_predictions = torch.cat(all_predictions, dim=0)  # (total_batches * batch_size, max_len)
    all_targets = torch.cat(all_targets, dim=0)  # (total_batches * batch_size, target_sequence_length)

    # Calculate sequence-level accuracy
    accuracy = sequence_accuracy(all_predictions, all_targets, pad_token)
    accuracy_t = token_accuracy(all_predictions, all_targets, pad_token)

    print(f"Greedy Decode, Sequence-level Accuracy: {accuracy}, Token Acuracy: {accuracy_t}")
    return accuracy_t
'''

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
np.random.seed(111)
torch.manual_seed(111)
seeds = [111, 222, 1111, 2222, 11111]

'''
# Add dataset
print("Add Dataset")
(add_vocab, add_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(200000, 1024)
add_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
add_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

add_train_loader = DataLoader(add_dataset_train, batch_size=64, shuffle=True)
add_test_loader = DataLoader(add_dataset_test, batch_size=64, shuffle=False)

example_train = add_dataset_train.__getitem__(3)
print(f"Add train example from tokens: input {decode_example(example_train[0], add_vocab)}, output {decode_example(example_train[1], add_vocab)}")
print(f"Length of Add Vocabulary: {len(add_vocab)}")

# AddNeg dataset
print("AddNeg Dataset")
(addNeg_vocab, addNeg_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(200000, 1024, negativeProbability=0.25)
addNeg_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
addNeg_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

addNeg_train_loader = DataLoader(addNeg_dataset_train, batch_size=64, shuffle=True)
addNeg_test_loader = DataLoader(addNeg_dataset_test, batch_size=64, shuffle=False)

example_train = addNeg_dataset_train.__getitem__(3)
print(f"AddNeg train example from tokens: input {decode_example(example_train[0], addNeg_vocab)}, output {decode_example(example_train[1], addNeg_vocab)}")
print(f"Length of AddNeg Vocabulary: {len(addNeg_vocab)}")
'''
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

example_train = reverse_dataset_train.__getitem__(1)
print(f"Reverse train example from tokens: input {decode_example(example_train[0], reverse_vocab)}, output {decode_example(example_train[1], reverse_vocab)}")
print(f"Length of Reverse Vocabulary: {len(reverse_vocab)}")
'''
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

example_train = dup_dataset_train.__getitem__(1)
print(f"Dup train example from tokens: input {decode_example(example_train[0], dup_vocab)}, output {decode_example(example_train[1], dup_vocab)}")
print(f"Length of Dup Vocabulary: {len(dup_vocab)}")

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

example_train = cart_dataset_train.__getitem__(1)
print(f"Cart train example from tokens: input {decode_example(example_train[0], cart_vocab)}, output {decode_example(example_train[1], cart_vocab)}")
print(f"Length of Cart Vocabulary: {len(cart_vocab)}")

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

example_train = inters_dataset_train.__getitem__(0)
print(f"Inters train example from tokens: input {decode_example(example_train[0], inters_vocab)}, output {decode_example(example_train[1], inters_vocab)}")
print(f"Length of Inters Vocabulary: {len(inters_vocab)}")


# Learning rate scheduler
class CustomScheduler2(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        super(CustomScheduler, self).__init__(optimizer, last_epoch)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model = torch.tensor(self.d_model, dtype=torch.float32)

    def get_lr(self):
        # Get the current training step
        step = max(1, self._step_count)  # Ensure step is at least 1 to avoid division by zero

        # Calculate the learning rate according to the custom schedule
        arg1 = torch.rsqrt(torch.tensor(step, dtype=torch.float32))
        arg2 = step * (self.warmup_steps ** -1.5)

        return [torch.rsqrt(self.d_model) * min(arg1, arg2)]

class CustomScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        # Ensure that step is at least 1 to avoid division by zero issues
        step = max(step, 1)
        
        # Calculate the two components of the learning rate schedule
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        # Return the final learning rate scaling factor
        return (self.d_model ** -0.5) * min(arg1, arg2)

class TransformerLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(TransformerLRScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        # This function calculates the current learning rate factor
        step = max(step, 1)
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * min(arg1, arg2)
'''
class ScheduledOptim():
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
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


# Utility function to get results
def get_results_Table1(vocab, max_len_enc, max_len_dec, epochs, train_loader,
                       test_loader, test_loader1=None, test_loader2=None, repetitions=1, device='cuda'):
    accuracies_rep = []
    for i in range(repetitions):
        torch.manual_seed(seeds[i])
        accuracies = []

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

        optimizer = ScheduledOptim(torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),1,  64, 4000)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        acc_1 = []
        for epoch in range (epochs):
            train_losses = train_step(transformer_abs, optimizer, train_loader, epoch=epoch, criterion=criterion, device=device)
            accuracy = test(transformer_abs, test_loader, max_len_dec, max_len_enc, device=device)
            if test_loader1 is not None and test_loader2 is not None:
                accuracy1 = test(transformer_abs, test_loader1, max_len_dec, max_len_enc, device=device)
                accuracy2 = test(transformer_abs, test_loader2, max_len_dec, max_len_enc, device=device)
                accuracy = accuracy2 #np.mean([accuracy, accuracy1, accuracy2])
            acc_1.append(accuracy)
        accuracies.append(np.mean(acc_1))
        
        acc_2 = []
        for transformer in transformers_rel:
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=False)
            for epoch in range(epochs):
                train_losses = train_step(transformer, optimizer, train_loader,
                                    relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epoch=epoch, criterion=criterion, device=device)
                accuracy = test(transformer, test_loader, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                if test_loader1 is not None and test_loader2 is not None:
                    accuracy1 = test(transformer, test_loader1, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy2 = test(transformer, test_loader2, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy2 #np.mean([accuracy, accuracy1, accuracy2])
                acc_2.append(accuracy)
            accuracies.append(np.mean(acc_2))

        acc_3 = []
        for transformer in transformers_rel2:
            optimizer = ScheduledOptim(torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), 1, 64, 4000)
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
            enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(max_len_enc, max_len_dec,
                                                                                                    16, dec2enc_ids=True)
            for epoch in range(epochs):
                train_losses = train_step(transformer, optimizer, train_loader,
                                    relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epoch=epoch, criterion=criterion, device=device)
                accuracy = test(transformer, test_loader, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids) ,device=device)
                if test_loader1 is not None and test_loader2 is not None:
                    accuracy1 = test(transformer, test_loader1, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy2 = test(transformer, test_loader2, max_len_dec, max_len_enc, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
                    accuracy = accuracy2 #np.mean([accuracy, accuracy1, accuracy2])
                acc_3.append(accuracy)
            accuracies.append(np.mean(acc_3))
        
        accuracies_rep.append(accuracies)
        
    accuracies_sum = [0] * 7 # 7 is the number of different models trained in Table1 for each dataset
    for i in range(repetitions):
        for j in range(7):
            accuracies_sum[j] += accuracies_rep[i][j]

    accuracies_mean = [x / repetitions for x in accuracies_sum]

    return accuracies_mean
    

# Table 1
def table1(device):
    '''
    # Add dataset
    accuracies_add = get_results_Table1(add_vocab, 26, 14, 2, add_train_loader, add_test_loader, repetitions=5, device=device)
    
    # AddNeg dataset
    accuracies_addNeg = get_results_Table1(addNeg_vocab, 26, 14, 10, addNeg_train_loader, addNeg_test_loader, repetitions=5, device=device)
    '''
    # Reverse dataset
    accuracies_reverse = get_results_Table1(reverse_vocab, 25, 26, 2, reverse_train_loader, reverse_test_loader0,
                                            reverse_test_loader1, reverse_test_loader2, repetitions=5, device=device)
    '''
    # Dup dataset
    accuracies_dup = get_results_Table1(dup_vocab, 25, 50, 4, dup_train_loader, dup_test_loader0, dup_test_loader1,
                                        dup_test_loader2, repetitions=5, device=device)
    '''
    #TODO SCAN-I, SCAN-aj, PCFG-p, PCFG-s datasets

    accuracies_addNeg = [0] * 7
    accuracies_add = [0] * 7
    accuracies_dup = [0] * 7
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup

    '''
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    #scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    scheduler = TransformerLRScheduler(optimizer, d_model=64)
    train_losses = train(transformer_abs, optimizer, scheduler, add_train_loader, epochs=2, device=device)
    test_loss, accuracy = test(transformer_abs, add_test_loader, add_vocab, device=device)
    accuracy = test_transformer_with_accuracy(transformer_abs, add_test_loader, 0, 3, 14, device=device)
    accuracies_add.append(accuracy)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = TransformerLRScheduler(optimizer, d_model=64)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy = test_transformer_with_accuracy(transformer, add_test_loader, 0, 3, 14, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = TransformerLRScheduler(optimizer, d_model=64)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy = test_transformer_with_accuracy(transformer, add_test_loader, 0, 3, 14, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, addNeg_train_loader, epochs=10, device=device)
    #test_loss, accuracy = test(transformer_abs, addNeg_test_loader, addNeg_vocab, device=device)
    accuracy = test_transformer_with_accuracy(transformer_abs, addNeg_test_loader, 0, 3, 14, device=device)
    accuracies_addNeg.append(accuracy)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracies_addNeg.append(accuracy)

    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=10, device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, reverse_train_loader, epochs=2, device=device)
    #test_loss0, accuracy0 = test(transformer_abs, reverse_test_loader0, reverse_vocab, device=device)
    #test_loss1, accuracy1 = test(transformer_abs, reverse_test_loader1, reverse_vocab, device=device)
    #test_loss2, accuracy2 = test(transformer_abs, reverse_test_loader2, reverse_vocab, device=device)
    #test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy0 = test_transformer_with_accuracy(transformer_abs, reverse_test_loader0, 0, 3, 6, device=device)
    accuracy1 = test_transformer_with_accuracy(transformer_abs, reverse_test_loader1, 0, 3, 26, device=device)
    accuracy2 = test_transformer_with_accuracy(transformer_abs, reverse_test_loader2, 0, 3, 26, device=device)
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy2)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=2, device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, dup_train_loader, epochs=4, device=device)
    #test_loss0, accuracy0 = test(transformer_abs, dup_test_loader0, dup_vocab, device=device)
    #test_loss1, accuracy1 = test(transformer_abs, dup_test_loader1, dup_vocab, device=device)
    #test_loss2, accuracy2 = test(transformer_abs, dup_test_loader2, dup_vocab, device=device)
    #test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy0 = test_transformer_with_accuracy(transformer_abs, dup_test_loader0, 0, 3, 50, device=device)
    accuracy1 = test_transformer_with_accuracy(transformer_abs, dup_test_loader1, 0, 3, 50, device=device)
    accuracy2 = test_transformer_with_accuracy(transformer_abs, dup_test_loader2, 0, 3, 50, device=device)
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy2)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs, optimizer, scheduler, cart_train_loader, epochs=4, device=device)
    #test_loss0, accuracy0 = test(transformer_abs, cart_test_loader0, cart_vocab, device=device)
    #test_loss1, accuracy1 = test(transformer_abs, cart_test_loader1, cart_vocab, device=device)
    #test_loss2, accuracy2 = test(transformer_abs, cart_test_loader2, cart_vocab, device=device)
    #test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy0 = test_transformer_with_accuracy(transformer_abs, cart_test_loader0, 0, 3, cart_max_seq_length_dec, device=device)
    accuracy1 = test_transformer_with_accuracy(transformer_abs, cart_test_loader1, 0, 3, cart_max_seq_length_dec1, device=device)
    accuracy2 = test_transformer_with_accuracy(transformer_abs, cart_test_loader2, 0, 3, cart_max_seq_length_dec2, device=device)
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy2)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=4, device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)
    
    
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

    optimizer = torch.optim.Adam(transformer_abs.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = TransformerLRScheduler(optimizer, d_model=64)
    train_losses = train(transformer_abs, optimizer, scheduler, inters_train_loader, epochs=8, device=device)
    test_loss0, accuracy0 = test(transformer_abs, inters_test_loader0, inters_vocab, device=device)
    test_loss1, accuracy1 = test(transformer_abs, inters_test_loader1, inters_vocab, device=device)
    test_loss2, accuracy2 = test(transformer_abs, inters_test_loader2, inters_vocab, device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy0 = test_transformer_with_accuracy(transformer_abs, inters_test_loader0, 0, 3, inters_max_seq_length_dec, device=device)
    accuracy1 = test_transformer_with_accuracy(transformer_abs, inters_test_loader1, 0, 3, inters_max_seq_length_dec1, device=device)
    accuracy2 = test_transformer_with_accuracy(transformer_abs, inters_test_loader2, 0, 3, inters_max_seq_length_dec2, device=device)
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy2)
    
    for transformer in transformers_rel:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = TransformerLRScheduler(optimizer, d_model=64)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=False)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy0 = test_transformer_with_accuracy(transformer, inters_test_loader0, 0, 3, inters_max_seq_length_dec, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy1 = test_transformer_with_accuracy(transformer, inters_test_loader1, 0, 3, inters_max_seq_length_dec1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device )
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=False)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy2 = test_transformer_with_accuracy(transformer, inters_test_loader2, 0, 3, inters_max_seq_length_dec2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_rel2:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = TransformerLRScheduler(optimizer, d_model=64)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), epochs=8, device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy0 = test_transformer_with_accuracy(transformer, inters_test_loader0, 0, 3, inters_max_seq_length_dec, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=True)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy1 = test_transformer_with_accuracy(transformer, inters_test_loader1, 0, 3, inters_max_seq_length_dec1, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device )
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        accuracy2 = test_transformer_with_accuracy(transformer, inters_test_loader2, 0, 3, inters_max_seq_length_dec2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
      
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters
    


# Table 2
def table2(device):
    # Add
    accuracies_add = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(add_vocab), len(add_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, add_train_loader, epochs=2, src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, add_test_loader, add_vocab, src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
    accuracies_add.append(accuracy)


    # AddNeg
    accuracies_addNeg = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(addNeg_vocab), len(addNeg_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=26, max_seq_length_dec=14, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, addNeg_train_loader, epochs=10, src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_abs_C, addNeg_test_loader, addNeg_vocab, src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_relEB_C, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    test_loss, accuracy = test(transformer_rel2EB_C, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
    accuracies_addNeg.append(accuracy)


    # Reverse
    accuracies_reverse = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(reverse_vocab), len(reverse_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=26, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, reverse_train_loader, epochs=2, src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, reverse_test_loader0, reverse_vocab, src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, reverse_test_loader1, reverse_vocab, src_vocab_size=len(reverse_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, reverse_test_loader2, reverse_vocab, src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_reverse.append(accuracy)
    

    # Dup
    accuracies_dup = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(dup_vocab), len(dup_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=25, max_seq_length_dec=50, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, dup_train_loader, epochs=4, src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, dup_test_loader0, dup_vocab, src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, dup_test_loader1, dup_vocab, src_vocab_size=len(dup_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, dup_test_loader2, dup_vocab, src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_relEB_C, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_dup.append(accuracy)
    

    # Cart
    accuracies_cart = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(cart_vocab), len(cart_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=cart_max_seq_length_enc2, max_seq_length_dec=cart_max_seq_length_dec2, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, cart_train_loader, epochs=4, src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, cart_test_loader0, cart_vocab, src_vocab_size=len(cart_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, cart_test_loader1, cart_vocab, src_vocab_size=len(cart_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, cart_test_loader2, cart_vocab, src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_relEB_C, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_cart.append(accuracy)

    
    # Inters
    accuracies_inters = []
    transformer_abs_C = Transformer.ExtendedStdTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2).to(device)
    transformer_relEB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel-eb").to(device)
    transformer_rel2EB_C = Transformer.ExtendedTransformer2(len(inters_vocab), len(inters_vocab), d=64, h=4, l=2, f=256, max_seq_length_enc=inters_max_seq_length_enc2, max_seq_length_dec=inters_max_seq_length_dec2, attention="rel2-eb").to(device)

    optimizer = torch.optim.Adam(transformer_abs_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    train_losses = train(transformer_abs_C, optimizer, scheduler, inters_train_loader, epochs=8, src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_abs_C, inters_test_loader0, inters_vocab, src_vocab_size=len(inters_vocab), device=device)
    test_loss1, accuracy1 = test(transformer_abs_C, inters_test_loader1, inters_vocab, src_vocab_size=len(inters_vocab), device=device)
    test_loss2, accuracy2 = test(transformer_abs_C, inters_test_loader2, inters_vocab, src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    optimizer = torch.optim.Adam(transformer_relEB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=False)
    train_losses = train(transformer_relEB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_relEB_C, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_relEB_C, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=False)
    test_loss2, accuracy2 = test(transformer_relEB_C, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
    accuracies_inters.append(accuracy)

    optimizer = torch.optim.Adam(transformer_rel2EB_C.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
    scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
    train_losses = train(transformer_rel2EB_C, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss0, accuracy0 = test(transformer_rel2EB_C, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
    test_loss1, accuracy1 = test(transformer_rel2EB_C, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
    test_loss2, accuracy2 = test(transformer_rel2EB_C, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
    test_loss = np.mean([test_loss0, test_loss1, test_loss2])
    accuracy = np.mean([accuracy0, accuracy1, accuracy2])
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0,  reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        accuracies_add.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, add_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
        test_loss, accuracy = test(transformer, add_test_loader, add_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(add_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        accuracies_addNeg.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(26, 14, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, addNeg_train_loader, epochs=10, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
        test_loss, accuracy = test(transformer, addNeg_test_loader, addNeg_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(addNeg_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_reverse.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 18, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, reverse_train_loader, epochs=2, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, reverse_test_loader0, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, reverse_test_loader1, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 26, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, reverse_test_loader2, reverse_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(reverse_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_dup.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(17, 34, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, dup_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, dup_test_loader0, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        test_loss1, accuracy1 = test(transformer, dup_test_loader1, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(25, 50, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, dup_test_loader2, dup_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(dup_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_cart.append(accuracy)

    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc, cart_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, cart_train_loader, epochs=4, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, cart_test_loader0, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc1, cart_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, cart_test_loader1, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(cart_max_seq_length_enc2, cart_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, cart_test_loader2, cart_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(cart_vocab), device=device)
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
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=64, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    for transformer in transformers_large:
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9) # Stessi iperparametri degli autori
        scheduler = CustomScheduler(optimizer, d_model=128, warmup_steps=4000)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc, inters_max_seq_length_dec, 16, dec2enc_ids=True)
        train_losses = train(transformer, optimizer, scheduler, inters_train_loader, epochs=8, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss0, accuracy0 = test(transformer, inters_test_loader0, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc1, inters_max_seq_length_dec1, 16, dec2enc_ids=False)
        test_loss1, accuracy1 = test(transformer, inters_test_loader1, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids = Encoding.create_relative_ids(inters_max_seq_length_enc2, inters_max_seq_length_dec2, 16, dec2enc_ids=True)
        test_loss2, accuracy2 = test(transformer, inters_test_loader2, inters_vocab, relative_ids=(enc_relative_ids, dec_relative_ids1, dec2enc_relative_ids), src_vocab_size=len(inters_vocab), device=device)
        test_loss = np.mean([test_loss0, test_loss1, test_loss2])
        accuracy = np.mean([accuracy0, accuracy1, accuracy2])
        accuracies_inters.append(accuracy)
    
    return accuracies_add, accuracies_addNeg, accuracies_reverse, accuracies_dup, accuracies_cart, accuracies_inters
'''

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1 = table1(device)
    print(f"Table1, Add dataset accuracies: {accuracies_add1}, \nAddNeg dataset accuracies: {accuracies_addNeg1},")
    print(f"Accuracies Reverse dataset: {accuracies_reverse1}, \nAccuracies Dup dataset: {accuracies_dup1}")
    #accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1, accuracies_cart1, accuracies_inters1 = table1(device)
    #accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_cart2, accuracies_inters2 = table2(device)
    #accuracies_add3, accuracies_addNeg3, accuracies_reverse3, accuracies_dup3, accuracies_cart3, accuracies_inters3 = table3(device)
    #accuracies_add4, accuracies_addNeg4, accuracies_reverse4, accuracies_dup4, accuracies_cart4, accuracies_inters4 = table4(device)
    
    #print(f"Table 1, Add dataset accuracies: {accuracies_add1},\nAddNeg dataset accuracies: {accuracies_addNeg1},\nReverse dataset accuracies: {accuracies_reverse1},\nDup dataset accuracies: {accuracies_dup1},\nCart dataset accuracies: {accuracies_cart1},\nInters dataset accuracies: {accuracies_inters1}")
    #print(f"Table 2, Add dataset accuracies: {accuracies_add2},\nAddNeg dataset accuracies: {accuracies_addNeg2},\nReverse dataset accuracies: {accuracies_reverse2},\nDup dataset accuracies: {accuracies_dup2},\nCart dataset accuracies: {accuracies_cart2},\nInters dataset accuracies: {accuracies_inters2}")
    #print(f"Table 3, Add dataset accuracies: {accuracies_add3},\nAddNeg dataset accuracies: {accuracies_addNeg3},\nReverse dataset accuracies: {accuracies_reverse3},\nDup dataset accuracies: {accuracies_dup3},\nCart dataset accuracies: {accuracies_cart3},\nInters dataset accuracies: {accuracies_inters3}")
    #print(f"Table 4, Add dataset accuracies: {accuracies_add4},\nAddNeg dataset accuracies: {accuracies_addNeg4},\nReverse dataset accuracies: {accuracies_reverse4},\nDup dataset accuracies: {accuracies_dup4},\nCart dataset accuracies: {accuracies_cart4},\nInters dataset accuracies: {accuracies_inters4}")
    
    current_dir = os.path.dirname(__file__)
    filepath1 = os.path.join(current_dir, "table1.csv")
    #filepath2 = os.path.join(current_dir, "table2.csv")
    #filepath3 = os.path.join(current_dir, "table3.csv")
    #filepath4 = os.path.join(current_dir, "table4.csv")

    res1 = pd.DataFrame(data=[accuracies_add1, accuracies_addNeg1, accuracies_reverse1, accuracies_dup1], columns=["abs", "rel-e", "rel-b", "rel-eb", "rel2-e", "rel2-b", "rel2-eb"])
    res1.index = ["Add", "AddNeg", "Reverse", "Dup"]
    res1.to_csv(filepath1)
    '''
    res2 = pd.DataFrame(data=[accuracies_add2, accuracies_addNeg2, accuracies_reverse2, accuracies_dup2, accuracies_cart2, accuracies_inters2], columns=["abs-c", "rel-eb-c", "rel2-eb-c"])
    res2.index = ["Add", "AddNeg", "Reverse", "Dup", "Cart", "Inters"]
    res2.to_csv(filepath2)
    '''
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