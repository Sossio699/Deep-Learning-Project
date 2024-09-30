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

            if src_vocab_size is None:
                if relative_ids is None:
                    predictions = transformer(source, target)
                else:
                    predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2])
            else:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size)
            
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

            if src_vocab_size is None:
                if relative_ids is None:
                    predictions = transformer(source, target)
                else:
                    predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2])
            else:
                predictions = transformer(source, target, relative_ids[0], relative_ids[1], relative_ids[2], src_vocab_size)   
                     
            loss_val = F.cross_entropy(predictions.permute(0, 2, 1), target, reduction="none")
            test_losses.append(loss_val.detach().numpy())
            sl_accuracy = sequence_level_accuracy(predictions, target)
            sl_accuracies.append(sl_accuracy)
            
    return np.mean(test_losses), np.mean(sl_accuracies)


# Hyperparameters
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embedding = 384
n_head = 6
n_layer = 6
dropout = 0.2


# Datasets and Dataloaders
class CustomDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input = input_tensor
        self.target = target_tensor

    def __len__(self):
                return len(self.target)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]
'''
# Add dataset
(add_vocab, add_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(192, 128)
add_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
add_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

add_train_loader = DataLoader(add_dataset_train, batch_size=64, shuffle=True)
add_test_loader = DataLoader(add_dataset_test, batch_size=64, shuffle=False)

# AddNeg dataset
(addNeg_vocab, addNeg_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_addition_dataset(192, 128, negativeProbability=0.25)
addNeg_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
addNeg_dataset_test = CustomDataset(input_tensor_val_list, target_tensor_val_list)

addNeg_train_loader = DataLoader(addNeg_dataset_train, batch_size=64, shuffle=True)
addNeg_test_loader = DataLoader(addNeg_dataset_test, batch_size=64, shuffle=False)

# Reverse dataset
(reverse_vocab, reverse_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_reversing_dataset(192, 128)
reverse_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
reverse_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
reverse_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
reverse_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

reverse_train_loader = DataLoader(reverse_dataset_train, batch_size=64, shuffle=True)
reverse_test_loader0 = DataLoader(reverse_dataset_test0, batch_size=64, shuffle=False)
reverse_test_loader1 = DataLoader(reverse_dataset_test1, batch_size=64, shuffle=False)
reverse_test_loader2 = DataLoader(reverse_dataset_test2, batch_size=64, shuffle=False)
'''
# Dup dataset
(dup_vocab, dup_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_duplicating_dataset(192, 128)
dup_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
dup_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
dup_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
dup_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

dup_train_loader = DataLoader(dup_dataset_train, batch_size=64, shuffle=True)
dup_test_loader0 = DataLoader(dup_dataset_test0, batch_size=64, shuffle=False)
dup_test_loader1 = DataLoader(dup_dataset_test1, batch_size=64, shuffle=False)
dup_test_loader2 = DataLoader(dup_dataset_test2, batch_size=64, shuffle=False)
'''
# Cart dataset
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
(inters_vocab, inters_vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list) = Datasets.create_intersection_dataset(192, 128)
inters_dataset_train = CustomDataset(input_tensor_train, target_tensor_train)
inters_dataset_test0 = CustomDataset(input_tensor_val_list[0], target_tensor_val_list[0])
inters_dataset_test1 = CustomDataset(input_tensor_val_list[1], target_tensor_val_list[1])
inters_dataset_test2 = CustomDataset(input_tensor_val_list[2], target_tensor_val_list[2])

inters_train_loader = DataLoader(inters_dataset_train, batch_size=64, shuffle=True)
inters_test_loader0 = DataLoader(inters_dataset_test0, batch_size=64, shuffle=False)
inters_test_loader1 = DataLoader(inters_dataset_test1, batch_size=64, shuffle=False)
inters_test_loader2 = DataLoader(inters_dataset_test2, batch_size=64, shuffle=False)
'''

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
    

def main():
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