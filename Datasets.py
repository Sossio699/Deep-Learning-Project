# Imports
import random
import torch
import torch.nn.utils.rnn as rnn_utils


# General tokens (common to all datasets):
PAD_TOKEN = "[PAD]"
SEP_TOKEN = "[SEP]"
END_TOKEN = "[END]"
START_TOKEN = "[START]"
END_ITERATION_TOKEN = "[ENDITER]"

PAD_TOKEN_IDX = 0
SEP_TOKEN_IDX = 1
END_TOKEN_IDX = 2
START_TOKEN_IDX = 3
END_ITERATION_TOKEN_IDX = 4


# Utilities
def max_length(tensor):
    return max(len(t) for t in tensor)

# Translates a dataset to padded token ID tensors
def create_dataset_tensors(examples_in_raw, examples_out_raw, max_len_inp, max_len_targ, vocab_to_int):
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
    in_tensor = torch.stack(in_list)
    out_tensor = torch.stack(out_list)
    in_tensor = rnn_utils.pad_sequence(in_tensor, batch_first=True) #TODO fare padding esatto con maxlen
    out_tensor = rnn_utils.pad_sequence(out_tensor, batch_first=True)

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
    
    def create_examples(n, minlen, maxlen, leftpadding=0, addAlignmentTokens=False, negativeProbability=0.0):
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
    
    if negativeProbability > 0:
        vocab_to_int["-"] = len(vocab)
        vocab.append("-")
        vocab_to_int["#"] = len(vocab)
        vocab.append("#")

    train_examples_in_raw, train_examples_out_raw = create_examples(trainsize, 1, 8, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, 5, 6, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_examples_in_raw, test_examples_out_raw = create_examples(testsize, 6, 8, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, 9, 10, leftpadding=leftpadding, negativeProbability=negativeProbability)

    max_len_inp = max(max_length(train_examples_in_raw), max_length(test_easy_examples_in_raw), max_length(test_hard_examples_in_raw))
    max_len_targ = max(max_length(train_examples_out_raw), max_length(test_easy_examples_out_raw), max_length(test_hard_examples_out_raw))

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, max_len_inp, max_len_targ, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize],  max_len_inp, max_len_targ, vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, max_len_inp, max_len_targ, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_examples_in_raw, test_examples_out_raw, max_len_inp, max_len_targ, vocab_to_int)
    input_tensor_val3, target_tensor_val3 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, max_len_inp, max_len_targ, vocab_to_int)

    return (vocab, vocab_to_int,
          input_tensor_train, target_tensor_train,
          [input_tensor_val0, input_tensor_val1,
           input_tensor_val2, input_tensor_val3],
          [target_tensor_val0,target_tensor_val1,
           target_tensor_val2, target_tensor_val3])


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
        for i in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        return examples_in, examples_out

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, 
                    "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


# Duplication dataset
def create_duplicating_dataset(trainsize, testsize, trainmindigits=1, trainmaxdigits=16, testmindigits=17, testmaxdigits=24):

    def create_example(minlen, maxlen):
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        l1 = random.randint(minlen, maxlen)
        example_in = []
        n_duplications = 2

        for i in range(l1):
            example_in.append(random.choice(digits))
        example_out = []
        for i in range(n_duplications):
            example_out += example_in
        example_in += [END_TOKEN]
        example_out = [START_TOKEN] + example_out + [END_TOKEN]
        return example_in, example_out
    
    def create_examples(n, minlen, maxlen):
        examples_in = []
        examples_out = []
        for i in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        
        return examples_in, examples_out
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN,
              "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, 
                    START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, 
                    "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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
    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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
        for i in range(l1):
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
            example_out = [START_TOKEN, "false", "END_TOKEN"]
        
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
    
    train_in, train_out = create_examples(trainsize, trainminelements, trainmaxelements)
    test_in, test_out = create_examples(testsize, testminelements, testmaxelements)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


#TODO download required files
# Setup for SCAN datasets
def create_scan_dataset(train_filename, test_filename):
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX,
                     START_TOKEN: START_TOKEN_IDX, END_ITERATION_TOKEN: END_ITERATION_TOKEN_IDX}

    def create_dataset_tensors_scan(instances, maxlen_inp=None, maxlen_targ=None):
        in_tensor = [] #TODO convertire in tensore?
        out_tensor = []
        for instance in instances:
            # Keep only the first and last steps (ignore intermediate steps):
            in_tensor.append(instance[0])
            out_tensor.append(instance[-1])

        #TODO controllare che sia giusto
        in_tensor = rnn_utils.pad_sequence(in_tensor, batch_first=True)
        out_tensor = rnn_utils.pad_sequence(out_tensor, batch_first=True)

        return in_tensor, out_tensor
    
    def tokenize_scan_line(line, vocab, vocab_to_int):
        instance_in_split = line.split(" ")

        # Find the tokens:
        for token in instance_in_split:
            if token not in vocab_to_int:
                vocab_to_int[token] = len(vocab)
                vocab.append(token)
        
        # Tokenize:
        instance_in_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_in_split] + [END_TOKEN_IDX])

        return instance_in_tokenized
    
    def load_and_tokenize_data(filename):
        instances_raw = []
        instances = []
        lines = 0 #TODO read files and modify this (guarda codice autori)
        for line in lines:
            if line.startswith("IN:"):
                line = line[4:]
                instance_raw = line.split(" OUT: ")
                instance = [tokenize_scan_line(instance_raw[0], vocab, vocab_to_int), tokenize_scan_line(instance_raw[1], vocab, vocab_to_int)]
                instances_raw.append(instance_raw)
                instances.append(instance)
        
        print("# instances: " + str(len(instances)))
        return instances_raw, instances
    
    instances_train_raw, instances_train = load_and_tokenize_data(train_filename)
    instances_test_raw, instances_test = load_and_tokenize_data(test_filename)

    input_tensor_train, target_tensor_train = create_dataset_tensors_scan(instances_train)
    input_tensor_val, target_tensor_val = create_dataset_tensors_scan(instances_test)
    max_length_train = max_length(input_tensor_train) #TODO capire come usarle
    max_length_val = max_length(input_tensor_val)
    max_length_targ_train = max_length(target_tensor_train)
    max_length_targ_val = max_length(target_tensor_val)

    testsize = len(instances_test)
    max_len_inp = max(max_length_train, max_length_val)
    max_len_targ = max(max_length_targ_train, max_length_targ_val)
    input_tensor_train, target_tensor_train = create_dataset_tensors_scan(instances_train, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors_scan(instances_train[0:testsize], maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val, target_tensor_val = create_dataset_tensors_scan(instances_test, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train,
             [input_tensor_val0, input_tensor_val], [target_tensor_val0, target_tensor_val])

# SCAN-length dataet
def create_scan_length_dataset():
    return create_scan_dataset("tasks_train_length.txt", "tasks_test_length.txt")

# SCAN-add-jump dataset
def create_scan_add_jump_dataset():
    return create_scan_dataset("tasks_train_addprim_jump.txt", "tasks_test_addprim_jump.txt")


#TODO Download required files
MAX_TRAIN_LEN = 128
MAX_TEST_LEN = 256
# PCFG-productivity and PCFG-sistematicity datasets
def create_pcfg_dataset(pcfg_split):
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, END_ITERATION_TOKEN: END_ITERATION_TOKEN_IDX}

    def create_dataset_tensors_pcfg(instances, maxlen_inp=None, maxlen_targ=None):
        in_tensor = [] #TODO convertire in tensore?
        out_tensor = []
        for instance in instances:
            for i in range(len(instance) - 1):
                in_tensor.append(instance[i])
                out_tensor.append(instance[i + 1])
        
        in_tensor = rnn_utils.pad_sequence(in_tensor, batch_first=True)
        out_tensor = rnn_utils.pad_sequence(out_tensor, batch_first=True)

        return in_tensor, out_tensor
    
    def load_and_tokenize_data(filename, maxlen):
        max_in_len = 0
        max_out_len = 0
        instances_raw = []
        instances = []
        lines_in = 0 #TODO read file and modify this
        lines_out = 0 #TODO read file and modify this
        instance_raw = []
        instance = []
        for i in range(len(lines_in)):
            instance_raw = [lines_in[i], lines_out[i]]
            instances_raw.append(instance_raw)

            for instance_part in instance_raw:
                for token in instance_part.split(" "):
                    if token not in vocab_to_int:
                        vocab_to_int[token] = len(vocab)
                        vocab.append(token)
            
            # Tokenize:
            instance_in_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[1].split(" ")] + [END_TOKEN_IDX])
            instance_out_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[1].split(" ")] + [END_TOKEN_IDX])
            if len(instance_out_tokenized) > maxlen:
                continue
            instances.append([instance_in_tokenized, instance_out_tokenized])

            max_in_len = max(max_in_len, len(instance_in_tokenized))
            max_out_len = max(max_out_len, len(instance_out_tokenized))
        
        print("max_in_len: " + str(max_in_len))
        print("max_out_len" + str(max_out_len))
        return instances_raw, instances
    
    instances_train_raw, instances_train = load_and_tokenize_data("pcfg_" + pcfg_split + "_train", MAX_TRAIN_LEN)
    instances_test_raw, instances_test = load_and_tokenize_data("pcfg_" + pcfg_split + "_test", MAX_TEST_LEN)

    input_tensor_train, target_tensor_train = create_dataset_tensors_pcfg(instances_train)
    input_tensor_val, target_tensor_val = create_dataset_tensors_pcfg(instances_test)
    max_length_train = max_length(input_tensor_train) #TODO capire come usarle
    max_length_val = max_length(input_tensor_val)
    max_length_targ_train = max_length(target_tensor_train)
    max_length_targ_val = max_length(target_tensor_val)

    testset_size = len(instances_test)
    max_len_inp = max(max_length_train, max_length_val)
    max_len_targ = max(max_length_targ_train, max_length_targ_val)
    input_tensor_train, target_tensor_train = create_dataset_tensors_pcfg(instances_train, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors_pcfg(instances_train[0:testset_size], maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val, target_tensor_val = create_dataset_tensors_pcfg(instances_test, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, [input_tensor_val0, input_tensor_val], [target_tensor_val0, target_tensor_val])


#TODO download required files
# COGS dataset
def create_cogs_dataset(max_len=256):
    # Set 'max_len' to 512 to ensure even the longest instances fit. Otherwise, longer instances will be discarded.
    MAX_COGS_TRAIN_LEN = max_len
    MAX_COGS_TEST_LEN = max_len

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, END_ITERATION_TOKEN: END_ITERATION_TOKEN_IDX}

    def create_dataset_tensors_cogs(instances, maxlen_inp=None, maxlen_targ=None):
        in_tensor = [] #TODO convertire in tensore?
        out_tensor = []
        for instance in instances:
            for i in range(len(instance) - 1):
                in_tensor.append(instance[i])
                out_tensor.append(instance[i + 1])
        
        in_tensor = rnn_utils.pad_sequence(in_tensor, batch_first=True)
        out_tensor = rnn_utils.pad_sequence(out_tensor, batch_first=True)

        return in_tensor, out_tensor
    
    def load_and_tokenize_data_internal(lines, maxlen):
        max_in_len = 0
        max_out_len = 0
        instances_raw = []
        instances = []
        lines_in = []
        lines_out = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 3:
                lines_in.append(parts[0])
                lines_out.append(parts[1])
        instance_raw = []
        instance = []
        for i in range(len(lines_in)):
            instance_raw = [lines_in[i], lines_out[i]]
            instances_raw.append(instance_raw)

            for instance_part in instance_raw:
                for token in instance_part.split(" "):
                    if token not in vocab_to_int:
                        vocab_to_int[token] = len(vocab)
                        vocab.append(token)
            
            # Tokenize:
            instance_in_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[0].split(" ")] + [END_TOKEN_IDX])
            instance_out_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[1].split(" ")] + [END_TOKEN_IDX])
            if len(instance_out_tokenized) > maxlen:
                continue
            instances.append([instance_in_tokenized, instance_out_tokenized])

            max_in_len = max(max_in_len, len(instance_in_tokenized))
            max_out_len = max(max_out_len, len(instance_out_tokenized))
        
        print("max_in_len: " + str(max_in_len))
        print("max_out_len: " + str(max_out_len))
        return instances_raw, instances

    def load_and_tokenize_data_by_distribution(filename, maxlen):
        instances_raw = []
        instances = []
        lines = 0 #TODO read file and modify this
        distribution_names = []
        lines_by_distribution = {}
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 3:
                distribution = parts[2]
                if distribution in lines_by_distribution:
                    lines_by_distribution[distribution].append(line)
                else:
                    distribution_names.append(distribution)
                    lines_by_distribution[distribution] = [line]
        
        for distribution in distribution_names:
            print(f"{distribution}: {len(lines_by_distribution[distribution])}")
        
        instances_raw_by_distribution = []
        instances_by_distribution = []
        for distribution in distribution_names:
            instances_raw, instances = load_and_tokenize_data_internal(lines_by_distribution[distribution], maxlen)
            instances_raw_by_distribution.append(instances_raw)
            instances_by_distribution.append(instances)
        return instances_raw_by_distribution, instances_by_distribution
    
    def load_and_tokenize_data(filename, maxlen):
        lines = 0 #TODO read file and modify this
        return load_and_tokenize_data_internal(lines, maxlen)
    
    instances_train_raw, instances_train = load_and_tokenize_data("COGS_train", MAX_COGS_TRAIN_LEN)
    instances_test_raw, instances_test = load_and_tokenize_data("COGS_test", MAX_COGS_TEST_LEN)
    instances_gen_raw, instances_gen = load_and_tokenize_data("COGS_gen", MAX_COGS_TEST_LEN)
    instances_gen_raw_by_dist, instances_gen_by_dist = (load_and_tokenize_data_by_distribution("COGS_gen", MAX_COGS_TEST_LEN))

    instances_list = [instances_train, instances_train[:len(instances_test)], instances_test] + instances_gen_by_dist + [instances_gen]

    input_tensors = [] #TODO convertire in tensore?
    target_tensors = []
    max_len_inp = 0
    max_len_targ = 0
    for instances in instances_list:
        input_tensor, target_tensor = create_dataset_tensors_cogs(instances)
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
        max_len_inp = max(max_len_inp, max_length(input_tensor))
        max_len_targ = max(max_len_targ, max_length(input_tensor))

    input_tensors = [] #TODO convertire in tensore?
    target_tensors = []
    for instances in instances_list:
        input_tensor, target_tensor = create_dataset_tensors_cogs(instances, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
        max_len_inp = max(max_len_inp, max_length(input_tensor))
        max_len_targ = max(max_len_targ, max_length(input_tensor))

    return (vocab, vocab_to_int, input_tensors[0], target_tensors[0], input_tensors[1:], target_tensors[1:])


#TODO download required files
MAX_TRAIN_LEN=256
MAX_TEST_LEN = 256

def cfq_decompose_output(line):
    tokens = line.split(" ")
    prefix = ""
    postfix = ""
    triplets_text = ""
    state = 0
    for token in tokens:
        if state == 0:
            if token == "{":
                prefix += token + " "
                state = 1
            else:
                prefix += token + " "
        elif state == 1:
            if token == "}":
                postfix += token + " "
                state = 2
            else:
                triplets_text += token + " "
        else:
            postfix += token + " "
    triplets = triplets_text.strip().split(" . ")
    return prefix, triplets, postfix

def cfq_rewrite_cartesian(triplets):
    if not triplets:
        return triplets
    triplet = triplets[0]
    tokens = triplet.split(" ")
    if len(tokens) == 3 and tokens[1] != "a":
        relation = tokens[1]
        left_tokens = [tokens[0]]
        right_tokens = [tokens[2]]
        relation_pairs = [(tokens[0], tokens[2])]
        to_delete = [triplet]
        to_keep = []
        for triplet2 in triplets[1:]:
            tokens2 = triplet2.split(" ")
            if len(tokens2) == 3 and tokens2[1] == relation:
                relation_pairs.append(tokens2[0], tokens2[2])
                if tokens2[0] not in left_tokens:
                    left_tokens.append(tokens2[0])
                if tokens2[2] not in right_tokens:
                    right_tokens.append(tokens2[2])
                to_delete.append(triplet2)
            else:
                to_keep.append(triplet2)
        # See if it's a cartesian product:
        any_missing = False
        for left_token in left_tokens:
            for right_token in right_tokens:
                if (left_token, right_token) not in relation_pairs:
                    any_missing = True
                    break
            if any_missing:
                break
        if any_missing:
            return ["( " + tokens[0] + " ) ( " + relation + " ) ( " + tokens[2] + " )"] + cfq_rewrite_cartesian(triplets[1:])
        else:
            # We have a cartesian product
            new_triplet = ("( " + " ".join(left_tokens) + " ) ( " + relation + " ) ( " + " ".join(right_tokens) + " )")
            return [new_triplet] + cfq_rewrite_cartesian(to_keep)
    else:
        return [triplet] + cfq_rewrite_cartesian(triplets[1:])

def cfq_merge_cartesians(triplets):
    if not triplets:
        return triplets
    triplet = triplets[0]
    if triplet[0] == "(":
        tokens = triplet.split(" ) ( ")
        if len(tokens) == 3:
            to_keep = []
            relations = [tokens[1]]
            for triplet2 in triplets[1:]:
                if triplet2[0] == "(":
                    tokens2 = triplet2.split(" ) ( ")
                    if (len(tokens2) == 3 and tokens[0] == tokens2[0] and tokens[2] == tokens2[2]):
                        relations.append(tokens2[1])
                    else:
                        to_keep.append(triplet2)
                else:
                    to_keep.append(triplet2)
            new_triplet = (tokens[0] + " ) ( " + " ".join(relations) + " ) ( " + tokens[2])
            return [new_triplet] + cfq_merge_cartesians(to_keep)
        else:
            return [triplet] + cfq_merge_cartesians(triplets[1:])
    else:
        return [triplet] + cfq_merge_cartesians(triplets[1:])

def simplify_cfq_output(output):
    prefix, triplets, postfix = cfq_decompose_output(output)
    triplets = cfq_rewrite_cartesian(triplets)
    triplets = cfq_merge_cartesians(triplets)
    return prefix + " . ".join(triplets) + " " + postfix

# CFQ-mcd1 dataset
def create_cfq_mcd1_dataset(simplify_cartesians=False):
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, END_ITERATION_TOKEN: END_ITERATION_TOKEN_IDX}

    def create_dataset_tensors_cfq(instances, maxlen_inp=None, maxlen_targ=None):
        in_tensor = [] #TODO convertire in tensore?
        out_tensor = []
        for instance in instances:
            for i in range(len(instance) - 1):
                in_tensor.append(instance[i])
                out_tensor.append(instance[i + 1])
        
        in_tensor = rnn_utils.pad_sequence(in_tensor, batch_first=True)
        out_tensor = rnn_utils.pad_sequence(out_tensor, batch_first=True)

        return in_tensor, out_tensor
    
    def load_and_tokenize_data(filename, maxlen):
        max_in_len = 0
        max_out_len = 0
        instances_raw = []
        instances = []
        lines = 0 #TODO read file and modify this
        lines_in = []
        lines_out = []
        for i in range(len(lines) // 2):
            lines_in.append(lines[i * 2].strip().replace("INPUT: ", ""))
            lines_out.append(lines[i * 2 + 1].strip().replace("OUTPUT: ", ""))
        instance_raw = []
        instance = []
        for i in range(len(lines_in)):
            instance_raw = [lines_in[i], lines_out[i]]
            if simplify_cartesians:
                instance_raw[1] = simplify_cfq_output(lines_out[i])
            instances_raw.append(instance_raw)

            for instance_part in instance_raw:
                for token in instance_part.split(" "):
                    if token not in vocab_to_int:
                        vocab_to_int[token] = len(vocab)
                        vocab.append(token)
            
            # Tokenize:
            instance_in_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[0].split(" ")] + [END_TOKEN_IDX])
            instance_out_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_raw[1].split(" ")] + [END_TOKEN_IDX])
            if len(instance_out_tokenized) > maxlen:
                continue
            instances.append([instance_in_tokenized, instance_out_tokenized])

            max_in_len = max(max_in_len, len(instance_in_tokenized))
            max_out_len = max(max_out_len, len(instance_out_tokenized))
        
        print("max_in_len: " + str(max_in_len))
        print("max_out_len: " + str(max_out_len))
        return instances_raw, instances
    
    instances_train_raw, instances_train = load_and_tokenize_data("cfq_mcd1_train", MAX_TRAIN_LEN)
    instances_test_raw, instances_test = load_and_tokenize_data("cfq_mcd1_dev", MAX_TEST_LEN)
    instances_gen_raw, instances_gen = load_and_tokenize_data("cfq_mcd1_test", MAX_TEST_LEN)
    
    input_tensor_train, target_tensor_train = create_dataset_tensors_cfq(instances_train)
    input_tensor_val, target_tensor_val = create_dataset_tensors_cfq(instances_test)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors_cfq(instances_gen)
    max_length_train = max_length(input_tensor_train)
    max_length_val = max_length(input_tensor_val)
    max_length_val2 = max_length(input_tensor_val2)
    max_length_targ_train = max_length(target_tensor_train)
    max_length_targ_val = max_length(target_tensor_val)
    max_length_targ_val2 = max_length(target_tensor_val2)

    #TODO controllare di non aver messo input di troppo alle funzioni
    testset_size = len(instances_test)
    max_len_inp = max(max_length_train, max_length_val, max_length_val2)
    max_len_targ = max(max_length_targ_train, max_length_targ_val, max_length_targ_val2)
    input_tensor_train, target_tensor_train = create_dataset_tensors_cfq(instances_train, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors_cfq(instances_train[0:testset_size], maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val, target_tensor_val = create_dataset_tensors_cfq(instances_test, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors_cfq(instances_gen, maxlen_inp=max_len_inp, maxlen_targ=max_len_targ)

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, [input_tensor_val0, input_tensor_val, input_tensor_val2], [target_tensor_val0, target_tensor_val, target_tensor_val2])