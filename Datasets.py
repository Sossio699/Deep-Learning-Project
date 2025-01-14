import random
import torch
import torch.nn.functional as F
from tensorflow import compat as ttf
tf=ttf.v1


# General tokens (common to all datasets):
PAD_TOKEN = "[PAD]"
SEP_TOKEN = "[SEP]"
END_TOKEN = "[END]"
START_TOKEN = "[START]"
END_ITERATION_TOKEN="[ENDITER]"

PAD_TOKEN_IDX = 0
SEP_TOKEN_IDX = 1
END_TOKEN_IDX = 2
START_TOKEN_IDX = 3
END_ITERATION_TOKEN_IDX = 4


# Utility function:
def max_length(tensor_list):
  return max(len(t) for t in tensor_list)


# Translates a dataset to padded token ID tensors
def create_dataset_tensors(examples_in_raw, examples_out_raw, max_len_in, max_len_out, vocab_to_int):
    in_list = []
    for example in examples_in_raw:
        list = [vocab_to_int[x] for x in example]
        in_list.append(list)
    out_list = []
    for example in examples_out_raw:
        list = [vocab_to_int[x] for x in example]
        out_list.append(list)

    in_tensors = []
    out_tensors = []
    for tensor in in_list:
        pad_tensor = tensor + [0] * (max_len_in - len(tensor))
        in_tensors.append(pad_tensor)
    for tensor in out_list:
        pad_tensor = tensor + [0] * (max_len_out - len(tensor))
        out_tensors.append(pad_tensor)
    
    return in_tensors, out_tensors

def create_dataset_tensors2(examples_in_raw, examples_out_raw, max_len_in, max_len_out, vocab_to_int):
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

    in_tensors = []
    out_tensors = []
    for tensor in in_list:
        pad_tensor = F.pad(tensor, (0, max_len_in - tensor.shape[-1]), "constant", 0)
        in_tensors.append(pad_tensor)
    for tensor in out_list:
        pad_tensor = F.pad(tensor, (0, max_len_out - tensor.shape[-1]), "constant", 0)
        out_tensors.append(pad_tensor)
    in_tensor = torch.stack(in_tensors, dim=0)
    out_tensor = torch.stack(out_tensors, dim=0)

    print(f"In tensor shape: {in_tensor.shape}")
    print(f"Out tensor shape: {out_tensor.shape}")

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
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, 5, 6, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_examples_in_raw, test_examples_out_raw = create_examples(testsize, 6, 8, leftpadding=leftpadding, negativeProbability=negativeProbability)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, 9, 10, leftpadding=leftpadding, negativeProbability=negativeProbability)

    max_len_inp = max(max_length(train_examples_in_raw), max_length(test_easy_examples_in_raw), max_length(test_hard_examples_in_raw))
    max_len_trg = max(max_length(train_examples_out_raw), max_length(test_easy_examples_out_raw), max_length(test_hard_examples_out_raw))

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_examples_in_raw, test_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val3, target_tensor_val3 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2, input_tensor_val3]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2, target_tensor_val3]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list, max_len_inp, max_len_trg)


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
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainmindigits, trainmaxdigits)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testmindigits, testmaxdigits)

    max_len_inp = max(max_length(train_examples_in_raw), max_length(test_easy_examples_in_raw), max_length(test_hard_examples_in_raw))
    max_len_trg = max(max_length(train_examples_out_raw), max_length(test_easy_examples_out_raw), max_length(test_hard_examples_out_raw))

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list, max_len_inp, max_len_trg)


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
    test_easy_examples_in_raw, test_easy_examples_out_raw = create_examples(testsize, trainmindigits, trainmaxdigits)
    test_hard_examples_in_raw, test_hard_examples_out_raw = create_examples(testsize, testmindigits, testmaxdigits)

    max_len_inp = max(max_length(train_examples_in_raw), max_length(test_easy_examples_in_raw), max_length(test_hard_examples_in_raw))
    max_len_trg = max(max_length(train_examples_out_raw), max_length(test_easy_examples_out_raw), max_length(test_hard_examples_out_raw))

    input_tensor_train, target_tensor_train = create_dataset_tensors(train_examples_in_raw, train_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val0, target_tensor_val0 = create_dataset_tensors(train_examples_in_raw[0:testsize], train_examples_out_raw[0:testsize], max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val1, target_tensor_val1 = create_dataset_tensors(test_easy_examples_in_raw, test_easy_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)
    input_tensor_val2, target_tensor_val2 = create_dataset_tensors(test_hard_examples_in_raw, test_hard_examples_out_raw, max_len_inp, max_len_trg, vocab_to_int)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val1, input_tensor_val2]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val1, target_tensor_val2]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list, max_len_inp, max_len_trg)


# SCAN dataset
def create_scan_dataset(train_filename, test_filename):
    # Preload the files:
    uploaded = {}
    to_download = [["tasks_train_length.txt", "Datasets/SCAN/tasks_train_length.txt"],
                   ["tasks_test_length.txt", "Datasets/SCAN/tasks_test_length.txt"],
                   ["tasks_train_addprim_jump.txt", "Datasets/SCAN/tasks_train_addprim_jump.txt"],
                   ["tasks_test_addprim_jump.txt", "Datasets/SCAN/tasks_test_addprim_jump.txt"],]
    for [name, path] in to_download:
        with tf.io.gfile.GFile(path, "rb") as f:
            uploaded[name] = f.read()
            lines = uploaded[name].decode("utf-8").split("\n")

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN:0, SEP_TOKEN:SEP_TOKEN_IDX, END_TOKEN:END_TOKEN_IDX,
                    START_TOKEN:START_TOKEN_IDX, END_ITERATION_TOKEN:END_ITERATION_TOKEN_IDX}
    
    def create_dataset_tensors_scan(instances, max_len_inp=None, max_len_targ=None):
        in_tensor_list = []
        out_tensor_list = []
        for instance in instances:
            # keep only the first and last steps (ignore intermediate steps)
            in_tensor_list.append(instance[0])
            out_tensor_list.append(instance[-1])
        
        max_len_input = max_length(in_tensor_list)
        max_len_target = max_length(out_tensor_list)

        in_pad_tensors = []
        out_pad_tensors = []
        for tensor in in_tensor_list:
            if max_len_inp is not None:
                pad_tensor = tensor + [0] * (max_len_inp - len(tensor))
            else:
                pad_tensor = tensor + [0] * (max_len_input - len(tensor))
            in_pad_tensors.append(pad_tensor)
        for tensor in out_tensor_list:
            if max_len_targ is not None:
                pad_tensor = tensor + [0] * (max_len_targ - len(tensor))
            else:
                pad_tensor = tensor + [0] * (max_len_target - len(tensor))
            out_pad_tensors.append(pad_tensor)

        return in_pad_tensors, out_pad_tensors, max_len_input, max_len_target

    def create_dataset_tensors_scan2(instances, max_len_inp = None, max_len_targ = None):
        in_tensor_list = []
        out_tensor_list = []
        for instance in instances:
            # keep only the first and last steps (ignore intermediate steps)
            in_tensor_list.append(instance[0])
            out_tensor_list.append(instance[-1])
        
        max_len_input = max_length(in_tensor_list)
        max_len_target = max_length(out_tensor_list)

        in_tensors = []
        out_tensors = []
        for instance in instances:
            in_tensors.append(torch.as_tensor(instance[0]))
            out_tensors.append(torch.as_tensor(instance[-1]))
        
        in_pad_tensors = []
        out_pad_tensors = []
        for tensor in in_tensors:
            if max_len_inp is not None:
                pad_tensor = F.pad(tensor, (0, max_len_inp - tensor.shape[-1]), "constant", 0)
            else:
                pad_tensor = F.pad(tensor, (0, max_len_input - tensor.shape[-1]), "constant", 0)
            in_pad_tensors.append(pad_tensor)
        for tensor in out_tensors:
            if max_len_targ is not None:
                pad_tensor = F.pad(tensor, (0, max_len_targ - tensor.shape[-1]), "constant", 0)
            else:
                pad_tensor = F.pad(tensor, (0, max_len_target - tensor.shape[-1]), "constant", 0)
            out_pad_tensors.append(pad_tensor)
    
        return in_pad_tensors, out_pad_tensors, max_len_input, max_len_target
    
    def tokenize_scan_line(line, vocab, vocab_to_int):
        instance_in_split = line.split(" ")

        # find the tokens:
        for token in instance_in_split:
            if token not in vocab_to_int:
                vocab_to_int[token] = len(vocab)
                vocab.append(token)

        # tokenize:
        instance_in_tokenized = ([START_TOKEN_IDX] + [vocab_to_int[x] for x in instance_in_split] + [END_TOKEN_IDX])
        return instance_in_tokenized
    
    def load_and_tokenize_data(filename):
        instances_raw = []
        instances = []
        lines = uploaded[filename].decode("utf-8").split("\n")
        for line in lines:
            if line.startswith("IN:"):
                line = line[4:]
                instance_raw = line.split(" OUT: ")
                instance = [tokenize_scan_line(instance_raw[0], vocab, vocab_to_int),
                            tokenize_scan_line(instance_raw[1], vocab, vocab_to_int)]
                instances_raw.append(instance_raw)
                instances.append(instance)

        print("# instances: " + str(len(instances)))
        return instances_raw, instances
    
    instances_train_raw, instances_train = load_and_tokenize_data(train_filename)
    instances_test_raw, instances_test = load_and_tokenize_data(test_filename)

    input_tensor_train, target_tensor_train, max_len_inp_train, max_len_targ_train = create_dataset_tensors_scan(instances_train)
    input_tensor_val, target_tensor_val, max_len_inp_test, max_len_targ_test = create_dataset_tensors_scan(instances_test)

    max_len_inp = max(max_len_inp_train, max_len_inp_test)
    max_len_targ = max(max_len_targ_train, max_len_targ_test)

    testsize = len(instances_test)
    input_tensor_train, target_tensor_train, _, _ = create_dataset_tensors_scan(instances_train, max_len_inp=max_len_inp, max_len_targ=max_len_targ)
    input_tensor_val0, target_tensor_val0, _, _ = create_dataset_tensors_scan(instances_train[0:testsize], max_len_inp=max_len_inp, max_len_targ=max_len_targ)
    input_tensor_val, target_tensor_val, _, _ = create_dataset_tensors_scan(instances_test, max_len_inp=max_len_inp, max_len_targ=max_len_targ)
    
    input_tensor_val_list = [input_tensor_val0, input_tensor_val]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list, max_len_inp, max_len_targ)


# PCFG dataset
def create_pcfg_datset(pcfg_split):
    MAX_TRAIN_LEN = 128
    MAX_TEST_LEN = 256

    uploaded_pcfg = {}
    to_download_pcfg = [["pcfg_productivity_train", "Datasets/PCFG/Productivity/train.src", "Datasets/PCFG/Productivity/train.tgt"],
                        ["pcfg_productivity_test", "Datasets/PCFG/Productivity/test.src", "Datasets/PCFG/Productivity/test.tgt"],
                        ["pcfg_systematicity_train", "Datasets/PCFG/Systematicity/train.src", "Datasets/PCFG/Systematicity/train.tgt"],
                        ["pcfg_systematicity_test", "Datasets/PCFG/Systematicity/test.src", "Datasets/PCFG/Systematicity/test.tgt"]]
    for [name, pathin, pathout] in to_download_pcfg:
        with tf.io.gfile.GFile(pathin, "rb") as fin:
            with tf.io.gfile.GFile(pathout, "rb") as fout:
                uploaded_pcfg[name] = [fin.read(), fout.read()]
                lines = uploaded_pcfg[name][0].decode("utf-8").split("\n")

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, END_ITERATION_TOKEN]
    vocab_to_int = {PAD_TOKEN:0, SEP_TOKEN:SEP_TOKEN_IDX, END_TOKEN:END_TOKEN_IDX, START_TOKEN:START_TOKEN_IDX, END_ITERATION_TOKEN:END_ITERATION_TOKEN_IDX}

    def create_dataset_tensors_pcfg(instances, max_len_inp=None, max_len_targ=None):
        in_tensor_list = []
        out_tensor_list = []
        for instance in instances:
            for i in range(len(instance) - 1):
                in_tensor_list.append(instance[i])
                out_tensor_list.append(instance[i + 1])
        
        max_len_input = max_length(in_tensor_list)
        max_len_target = max_length(out_tensor_list)                
        
        in_pad_tensors = []
        out_pad_tensors = []
        for tensor in in_tensor_list:
            if max_len_inp is not None:
                pad_tensor = tensor + [0] * (max_len_inp - len(tensor))
            else:
                pad_tensor = tensor + [0] * (max_len_input - len(tensor))
            in_pad_tensors.append(pad_tensor)
        for tensor in out_tensor_list:
            if max_len_targ is not None:
                pad_tensor = tensor + [0] * (max_len_targ - len(tensor))
            else:
                pad_tensor = tensor + [0] * (max_len_target - len(tensor))
            out_pad_tensors.append(pad_tensor)
    
        return in_pad_tensors, out_pad_tensors, max_len_input, max_len_target

    def create_dataset_tensors_pcfg2(instances, max_len_inp=None, max_len_targ=None):
        in_tensor_list = []
        out_tensor_list = []
        for instance in instances:
            for i in range(len(instance) - 1):
                in_tensor_list.append(instance[i])
                out_tensor_list.append(instance[i + 1])
        
        max_len_input = max_length(in_tensor_list)
        max_len_target = max_length(out_tensor_list)

        in_tensors = []
        out_tensors = []
        for instance in instances:
            in_instance = []
            out_instance = []
            for i in range(len(instance) - 1):
                in_tensors.append(torch.as_tensor(instance[i]))
                out_tensors.append(torch.as_tensor(instance[i + 1]))
                #in_instance.append(instance[i]) TODO scegliere quale delle due usare
                #out_instance.append(instance[i + 1])
            #in_tensors.append(torch.as_tensor(in_instance))
            #out_tensors.append(torch.as_tensor(out_instance))
                
        
        in_pad_tensors = []
        out_pad_tensors = []
        for tensor in in_tensors:
            if max_len_inp is not None:
                pad_tensor = F.pad(tensor, (0, max_len_inp - tensor.shape[-1]), "constant", 0)
            else:
                pad_tensor = F.pad(tensor, (0, max_len_input - tensor.shape[-1]), "constant", 0)
            in_pad_tensors.append(pad_tensor)
        for tensor in out_tensors:
            if max_len_targ is not None:
                pad_tensor = F.pad(tensor, (0, max_len_targ - tensor.shape[-1]), "constant", 0)
            else:
                pad_tensor = F.pad(tensor, (0, max_len_target - tensor.shape[-1]), "constant", 0)
            out_pad_tensors.append(pad_tensor)
    
        return in_pad_tensors, out_pad_tensors, max_len_input, max_len_target
    
    def load_and_tokenize_data(filename, maxlen):
        max_in_len = 0
        max_out_len = 0
        instances_raw = []
        instances = []
        lines_in = uploaded_pcfg[filename][0].decode("utf-8").split("\n")
        lines_out = uploaded_pcfg[filename][1].decode("utf-8").split("\n")
        instance_raw = []
        for i in range(len(lines_in)):
            instance_raw = [lines_in[i], lines_out[i]]
            instances_raw.append(instance_raw)

            for instance_part in instance_raw:
                for token in instance_part.split(" "):
                    if token not in vocab_to_int:
                        vocab_to_int[token] = len(vocab)
                        vocab.append(token)

            # tokenize:
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
    
    instances_train_raw, instances_train = load_and_tokenize_data("pcfg_" + pcfg_split + "_train", MAX_TRAIN_LEN)
    instances_test_raw, instances_test = load_and_tokenize_data("pcfg_" + pcfg_split + "_test", MAX_TEST_LEN)

    input_tensor_train, target_tensor_train, max_len_inp_train, max_len_targ_train = create_dataset_tensors_pcfg(instances_train)
    input_tensor_val, target_tensor_val, max_len_inp_test, max_len_targ_test = create_dataset_tensors_pcfg(instances_test)

    max_len_inp = max(max_len_inp_train, max_len_inp_test)
    max_len_targ = max(max_len_targ_train, max_len_targ_test)

    testset_size = len(instances_test)
    input_tensor_train, target_tensor_train, _, _ = create_dataset_tensors_pcfg(instances_train, max_len_inp=max_len_inp, max_len_targ=max_len_targ)
    input_tensor_val0, target_tensor_val0, _, _ = create_dataset_tensors_pcfg(instances_train[0:testset_size], max_len_inp=max_len_inp,max_len_targ=max_len_targ)
    input_tensor_val, target_tensor_val, _, _ = create_dataset_tensors_pcfg(instances_test, max_len_inp=max_len_inp, max_len_targ=max_len_targ)

    input_tensor_val_list = [input_tensor_val0, input_tensor_val]
    target_tensor_val_list = [target_tensor_val0, target_tensor_val]

    return (vocab, vocab_to_int, input_tensor_train, target_tensor_train, input_tensor_val_list, target_tensor_val_list, max_len_inp, max_len_targ)