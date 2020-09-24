import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """

    corpus = {}
    # Getting the list of all the files in directory
    pathDir = os.listdir(os.path.join(os.path.curdir, directory))
    os.chdir(directory)
    for filename in pathDir:
        # Reading the words in filename and storing them in corpus dictionary
        with open(filename, encoding="utf8") as openFile:
            corpus[filename] = openFile.read()
        openFile.close()
    return corpus

    raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    # Tokenizing the document
    words = nltk.word_tokenize(document)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    final_words = [word.lower() for word in words]
    # Filtering out all the stop words and punctuations from words
    final_words = [word for word in final_words if word not in stop_words and word not in string.punctuation]

    return final_words

    raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.

    """

    idfs = dict()
    words = set()
    # Forming a set of all the words present in the documents
    for filename in documents:
        words.update(documents[filename])

    # Calculating IDF as all the words present in each files
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs

    raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.

    """

    file_scores = dict()
    # Calculating the tf-idf of the files
    for filename in files:
        tf_idfs = dict()
        for word in query:
            if word in files[filename]:
                count = files[filename].count(word)
                tf_idfs[word] = count * idfs[word]
        scores = tf_idfs.values()
        file_scores[filename] = sum(scores)

    # Sorting the files according to the tf-idfs in descending order
    file_scores = {k: v for k, v in sorted(file_scores.items(), key=lambda item: item[1], reverse=True)}
    file_list = []
    for file in file_scores:
        file_list.append(file)

    # Selecting the top n files from the list
    file_list = file_list[:n]

    return file_list

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """

    # Getting the list of sentences and their total idfs and also their quality term density
    sentence_scores = dict()
    qtd = dict()
    for sentence in sentences:
        sentence_scores[sentence] = 0
        count = 0
        for word in query:
            if word in sentences[sentence]:
                sentence_scores[sentence] += idfs[word]
                count += 1
        qtd[sentence] = count / len(sentences[sentence])

    # Sorting the list of sentence in descending order
    rank_sentences = sorted(sentence_scores.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

    # Checking for Quality term density in case of tie between total idfs of sentences
    for i in range(len(rank_sentences) - 1):
        if rank_sentences[i][1] == rank_sentences[i+1][1]:
            if qtd[rank_sentences[i][0]] < qtd[rank_sentences[i+1][0]]:
                temp = rank_sentences[i]
                rank_sentences[i] = rank_sentences[i + 1]
                rank_sentences[i + 1] = temp

    # Selecting top n sentences from the list
    rank_sentences = rank_sentences[:n]
    rank_list = []
    for i in range(len(rank_sentences)):
        rank_list.append(rank_sentences[i][0])

    return rank_list

    raise NotImplementedError


if __name__ == "__main__":
    main()
