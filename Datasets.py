# Imports
import random
import torch
import torch.nn.utils.rnn as rnn_utils

# Function to visualize examples better
def decode_example(example):
    output = ""
    for element in example:
        output += element +" "
    return output


# General tokens (common to all datasets):
PAD_TOKEN = "[PAD]"
SEP_TOKEN = "[SEP]"
END_TOKEN = "[END]"
START_TOKEN = "[START]"

PAD_TOKEN_IDX = 0
SEP_TOKEN_IDX = 1
END_TOKEN_IDX = 2
START_TOKEN_IDX = 3


# Utilities
def max_length(tensor):
    return max(len(t) for t in tensor)

# Translates a dataset to padded token ID tensors
def create_dataset_tensors(examples_in_raw, examples_out_raw, vocab_to_int):
    in_list = []
    for example in examples_in_raw:
        list = [vocab_to_int[x] for x in example]
        tensor = torch.as_tensor(list)
        in_list.append(tensor)
    out_list = []
    for example in examples_out_raw:
        list = [vocab_to_int[x] for x in example]
        tensor = torch.as_tensor(list)
        out_list.append(tensor)
        
    #in_tensor = torch.cat(in_list, dim=0)
    #out_tensor = torch.cat(out_list, dim=0)
    in_tensor = rnn_utils.pad_sequence(in_list, batch_first=True)
    out_tensor = rnn_utils.pad_sequence(out_list, batch_first=True)

    print(in_tensor.shape)
    print(out_tensor.shape)

    return in_tensor, out_tensor


# Functions to create datasets
# Addition and AdditionNegatives dataset
def create_addition_dataset(trainsize, testsize, leftpadding=12, negativeProbability=0.0):
    
    def create_example(minlen, maxlen, leftpadding=0, negativeProbability=0.0):
        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        digits1 = []
        digits2 = []
        digitssum = []
        int1 = 0
        int2 = 0
        l1 = random.randint(minlen, maxlen)
        l2 = random.randint(minlen, maxlen)
        for i in range(l1):
            n = random.choice(numbers)
            int1 = (int1 * 10) + n
            digits1.append(str(n))
        for i in range(l2):
            n = random.choice(numbers)
            int2 = (int2 * 10) + n
            digits2.append(str(n))
        if random.random() < negativeProbability:
            int1 = -int1
            digits1 = ["-"] + digits1
        if random.random() < negativeProbability:
            int2 = -int2
            digits2 = ["-"] + digits2
        
        sum = int1 + int2
        negatedSum = False
        sumtmp = sum
        if sumtmp < 0:
            negatedSum = True
            sumtmp = -sumtmp
        while sumtmp > 0:
            digitssum = [str(sumtmp % 10)] + digitssum
            sumtmp //= 10
        if negatedSum:
            digitssum = ["-"] + digitssum
        
        leftpaddingtoken = "0"
        if negativeProbability > 0:
            leftpaddingtoken = "#"
        
        while len(digits1) < leftpadding:
            digits1 = [leftpaddingtoken] + digits1
        while len(digits2) < leftpadding:
            digits2 = [leftpaddingtoken] + digits2
        while len(digitssum) < leftpadding:
            digitssum = [leftpaddingtoken] + digitssum
        
        example_in = digits1 + [SEP_TOKEN] + digits2 + [END_TOKEN]
        example_out = [START_TOKEN] + digitssum + [END_TOKEN]

        return example_in, example_out
    
    def create_examples(n, minlen, maxlen, leftpadding=0, negativeProbability=0.0):
        examples_in = []
        examples_out = []
        for _ in range(n):
            ein, eout = create_example(minlen, maxlen, leftpadding, negativeProbability)
            examples_in.append(ein)
            examples_out.append(eout)
        return examples_in, examples_out

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX,
                     START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6,
                       "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13} # Indici associati ai token
    
    #vocab_to_int["#"] = len(vocab)
    #vocab.append("#")
    if negativeProbability > 0:
        vocab_to_int["-"] = len(vocab)
        vocab.append("-")
        vocab_to_int["#"] = len(vocab)
        vocab.append("#")

    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, 1, 8, leftpadding=leftpadding, negativeProbability=negativeProbability)
    if negativeProbability > 0.0:
        print(f"AddNeg train example: input {decode_example(train_examples_in_raw[3])}, output {decode_example(train_examples_out_raw[3])}")
    else:
        print(f"Add train example: input {decode_example(train_examples_in_raw[3])}, output {decode_example(train_examples_out_raw[3])}")
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, 5, 6, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_examples_in_raw, test_examples_out_raw = create_examples(testsize, 6, 8, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, 9, 10, leftpadding=leftpadding, negativeProbability=negativeProbability)

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_examples_in_raw, test_examples_out_raw, vocab_to_int)
    input_tensor_val3, target_tensor_val3 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, vocab_to_int)

    input_tensor_val = torch.cat((input_tensor_val0, input_tensor_val1, input_tensor_val2, input_tensor_val3), dim=0)
    target_tensor_val = torch.cat((target_tensor_val0, target_tensor_val1, target_tensor_val2, target_tensor_val3), dim=0)

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val, target_tensor_val)


# Reversing dataset
def create_reversing_dataset(trainsize, testsize, trainmindigits=1, trainmaxdigits=16, testmindigits=17, testmaxdigits=24):

    def create_example(minlen, maxlen):
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        l1 = random.randint(minlen, maxlen)
        example_in = []
        for _ in range(l1):
            example_in.append(random.choice(digits))
        example_out = example_in[::-1]
        example_in.append(END_TOKEN)
        example_out = [START_TOKEN] + example_out + [END_TOKEN]
        return example_in, example_out
    
    def create_examples(n, minlen, maxlen):
        examples_in = []
        examples_out = []
        for _ in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        return examples_in, examples_out

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, 
                    "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, trainmindigits, trainmaxdigits)
    print(f"Reverse train example: input {decode_example(train_examples_in_raw[1])}, output {decode_example(train_examples_out_raw[1])}")
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainmindigits, trainmaxdigits)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testmindigits, testmaxdigits)

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list)


# Duplication dataset
def create_duplicating_dataset(trainsize, testsize, trainmindigits=1, trainmaxdigits=16, testmindigits=17, testmaxdigits=24):

    def create_example(minlen, maxlen):
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        l1 = random.randint(minlen, maxlen)
        example_in = []
        n_duplications = 2

        for _ in range(l1):
            example_in.append(random.choice(digits))
        example_out = []
        for _ in range(n_duplications):
            example_out += example_in
        example_in += [END_TOKEN]
        example_out = [START_TOKEN] + example_out + [END_TOKEN]
        return example_in, example_out
    
    def create_examples(n, minlen, maxlen):
        examples_in = []
        examples_out = []
        for _ in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        
        return examples_in, examples_out
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, 
                    "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, trainmindigits, trainmaxdigits)
    print(f"Dup train example: input {decode_example(train_examples_in_raw[1])}, output {decode_example(train_examples_out_raw[1])}")
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainmindigits, trainmaxdigits)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testmindigits, testmaxdigits)

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list)


# Cartesian dataset
def create_cartesian_dataset(trainsize, testsize, trainmindigits=1, trainmaxdigits=6, testmindigits=7, testmaxdigits=8):

    def create_example(minlen, maxlen):
        symbols1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        symbols2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        l1 = random.randint(minlen, maxlen)
        l2 = random.randint(minlen, maxlen)
        set1 = []
        set2 = []
        for _ in range(l1):
            set1.append(random.choice(symbols1))
        for _ in range(l2):
            set2.append(random.choice(symbols2))
        example_in = set1 + [SEP_TOKEN] + set2 + [END_TOKEN]
        example_out = []
        for i in set1:
            for j in set2:
                example_out.append(i)
                example_out.append(j)
                example_out.append(SEP_TOKEN)
        example_out.pop() # remove last SEP_TOKEN
        example_out.append(END_TOKEN)

        return example_in, [START_TOKEN] + example_out
    
    def create_examples(n, minlen, maxlen):
        examples_in = []
        examples_out = []
        for _ in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        
        return examples_in, examples_out
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6,
                      "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13, "a": 14, "b": 15,
                        "c": 16, "d": 17, "e": 18, "f": 19, "g": 20, "h": 21, "i": 22, "j": 23}
    
    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, trainmindigits, trainmaxdigits)
    print(f"Cart train example: input {decode_example(train_examples_in_raw[1])}, output {decode_example(train_examples_out_raw[1])}")
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainmindigits, trainmaxdigits)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testmindigits, testmaxdigits)

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list)


# Intersection dataset
def create_intersection_dataset(trainsize, testsize, trainminelements=1, trainmaxelements=16, testminelements=17, testmaxelements=24):
    elements = []
    for a in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]:
        for b in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            elements.append(a + b)
        
    def create_example(minlen, maxlen, label):
        l1 = random.randint(minlen, maxlen)
        l2 = random.randint(minlen, maxlen)
        example_in = []
        intersection = []
        set1 = []
        set2 = []
        for _ in range(l1):
            element = random.choice(elements)
            if element not in set1:
                set1.append(element)
        while len(set2) < l2:
            element = random.choice(elements)
            if element not in set2:
                if element in set1:
                    if label == "true":
                        intersection.append(element)
                        set2.append(element)
                else:
                    set2.append(element)
        if label == "true" and not intersection:
            element = random.choice(set1)
            set2[random.choice(range(len(set2)))] = element
            intersection.append(element)
        
        example_in = set1 + [SEP_TOKEN] + set2 + [END_TOKEN]
        if intersection:
            example_out = [START_TOKEN, "true", END_TOKEN]
        else:
            example_out = [START_TOKEN, "false", END_TOKEN]
        
        return example_in, example_out

    def create_examples(n, minlen, maxlen):
        examples_in = []
        examples_out = []
        n_positive = 0
        for i in range(n):
            ein, eout = create_example(minlen, maxlen, ["true", "false"][i % 2])
            examples_in.append(ein)
            examples_out.append(eout)
            if "true" in eout:
                n_positive +=1
        print(f"positive: {n_positive}, negative: {n - n_positive}")
        return examples_in, examples_out
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, "true", "false"] + elements
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "true": 4, "false": 5}
    for element in elements:
        vocab_to_int[element] = len(vocab_to_int)
    
    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, trainminelements, trainmaxelements)
    print(f"Inters train example: input {decode_example(train_examples_in_raw[0])}, output {decode_example(train_examples_out_raw[0])}")
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainminelements, trainmaxelements)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testminelements, testmaxelements)

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list)