import random
import copy

def grade_dictionary_class_expand_dictionary(dictionary, n = 50):
    for i in range(n):
        grade_dictionary_class_expand_dictionary_single(dictionary)

def grade_dictionary_class_expand_dictionary_single(dictionary):
    dictionary_before = copy.deepcopy(dictionary)
    random_words = (random.randint(0, len(dictionary) - 1), random.randint(0, len(dictionary) - 1))
    while random_words[0] == random_words[1]:
        random_words = (random.randint(0, len(dictionary) - 1), random.randint(0, len(dictionary) - 1))
    
    dictionary.expand_dictionary(random_words)
    assert len(dictionary) == len(dictionary_before) + 1, "Dictionary size is not correct"
    assert dictionary.dictionary_array == dictionary_before.dictionary_array + [dictionary_before.dictionary_array[random_words[0]] + dictionary_before.dictionary_array[random_words[1]]], "Dictionary array is not correct"
    assert dictionary.combinations_to_index[random_words] == len(dictionary) - 1, "Combinations to index is not correct"