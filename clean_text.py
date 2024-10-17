from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


import re


class clean_text:
    # Remove English stop Words from description
    def all_upper(stopwords):
        return [x.upper() for x in stopwords]


    def clearDesc_nltk(x, correct):
        # Lista das Stopwords em inglês
        stop_wordsEn = set(clean_text.all_upper(stopwords.words("english")))


        x = re.sub(r"[^a-z A-Z]+", "", x)


        # Separate the descriptions words
        tokenize = word_tokenize(x)
        tokenize = clean_text.all_upper(tokenize)
        filterSentence = ""


        # Go through the list of words searching for the stop words
        for w in tokenize:
            if w not in stop_wordsEn:
                filterSentence = filterSentence + " " + w


        return filterSentence


    def clearDescSimpl(x, correct):

        x = re.sub(r"[^a-z A-Z]+", "", x)

        return x


    def clearDesc_list(x, correct):
        # Lista das Stopwords em inglês
        # stop_wordsEn = set(clean_text.all_upper(stopwords.words('english')))

        x = re.sub(r"[^a-z A-Z]+", "", x)


        for line in correct.itertuples():
            if line[1] in x:
                x = re.sub(line[1], line[2], x)


        return x


    #
    def clearDesc_list_nltk(x, correct):
        # Lista das Stopwords em inglês
        stop_wordsEn = set(clean_text.all_upper(stopwords.words("english")))


        x = re.sub(r"[^a-z A-Z]+", "", x)


        # Separate the descriptions words
        tokenize = word_tokenize(x)
        tokenize = clean_text.all_upper(tokenize)
        filterSentence = ""


        # Go through the list of words searching for the stop words
        for w in tokenize:
            if w not in stop_wordsEn:
                filterSentence = filterSentence + " " + w


        for line in correct.itertuples():
            if line[1] in filterSentence:
                filterSentence = re.sub(line[1], line[2], filterSentence)


        return filterSentence
