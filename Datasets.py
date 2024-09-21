# Imports
import random


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
            ein, eout = create_example(minlen, maxlen, leftpadding, addAlignmentTokens, negativeProbability)
            examples_in.append(ein)
            examples_out.append(eout)
        return examples_in, examples_out

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}
    
    if negativeProbability > 0:
        vocab_to_int["-"] = len(vocab)
        vocab.append("-")
        vocab_to_int["#"] = len(vocab)
        vocab.append("#")
    
    train_in, train_out = create_examples(trainsize, 1, 8, leftpadding, negativeProbability)
    test_in, test_out = create_examples(testsize, 5, 6, leftpadding, negativeProbability)
    #TODO definire tensori come sul codice online

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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

    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13}

    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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
        for i in range(n):
            ein, eout = create_example(minlen, maxlen)
            examples_in.append(ein)
            examples_out.append(eout)
        
        return examples_in, examples_out
    
    vocab = [PAD_TOKEN, SEP_TOKEN, END_TOKEN, START_TOKEN, "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, "0": 4, "1": 5, "2": 6, "3": 7, "4": 8, "5": 9, "6": 10, "7": 11, "8": 12, "9": 13, "a": 14, "b": 15, "c": 16, "d": 17, "e": 18, "f": 19, "g": 20, "h": 21, "i": 22, "j": 23}
    train_in, train_out = create_examples(trainsize, trainmindigits, trainmaxdigits)
    test_in, test_out = create_examples(testsize, testmindigits, testmaxdigits)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)


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
    vocab_to_int = {PAD_TOKEN: 0, SEP_TOKEN: SEP_TOKEN_IDX, END_TOKEN: END_TOKEN_IDX, START_TOKEN: START_TOKEN_IDX, "true": 4, "false": 5}
    for element in elements:
        vocab_to_int[element] = len(vocab_to_int)
    
    train_in, train_out = create_examples(trainsize, trainminelements, trainmaxelements)
    test_in, test_out = create_examples(testsize, testminelements, testmaxelements)
    #TODO definire tensori

    return (vocab, vocab_to_int, train_in, train_out, test_in, test_out)