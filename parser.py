import nltk
import sys
import copy

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """

S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
AP -> Adj | Adj AP
AVP -> Adv | Adv AVP
NP -> N | Det AP N | Det N | PP NP | AP NP
PP -> P | P NP
VP -> V | V NP | VP NP | V NP PP AVP | AVP VP | V NP AVP | V AVP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenizing the word into a list
    words = nltk.word_tokenize(sentence, language="english")
    final_words = []
    for word in words:
        # Filtering out all non alphabets
        if word.isalpha():
            # Converting all words to lowercase
            final_words.append(word.lower())
    return final_words

    raise NotImplementedError

def contains_NP(tree):
    # Checking if a tree has NP as its subtree
    for subtree in tree:
        if subtree.label() == "NP":
            return True
    return False

def get_np_chunks(tree):

    # Getting all the np chunks in form to multi-dimensional list
    NP_chunks = []
    # Checking if the Node is leaf or not
    if len(list(tree.subtrees())) > 1:
        for subtree in tree:
            # Checking if tree is NP
            if subtree.label() == "NP":
                # Checking if the tree is NP chunk
                check = contains_NP(subtree)
                if not check:
                    NP_chunks.append(subtree)
                    continue
            NP_chunks.append(np_chunk(subtree))
    return NP_chunks

def clean_list(np_chunks_list, np_list):

    # Converting a multi-dimensional list into a list of trees
    for np in np_list:
        if type(np) == type(list()):
            clean_list(np_chunks_list, np)
        else:
            np_chunks_list.append(np)


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """

    # Getting all the NP chunks in multi-dimensional list
    np_chunks = get_np_chunks(tree)
    np_chunks_copy = copy.deepcopy(np_chunks)
    # Removing all the empty lists
    for np in np_chunks:
        if len(np) == 0:
            np_chunks_copy.remove(np)
    final_np_chunks = []
    # Converting a multi-dimensional list into a list of trees
    clean_list(final_np_chunks, np_chunks_copy)

    return final_np_chunks

    raise NotImplementedError


if __name__ == "__main__":
    main()
