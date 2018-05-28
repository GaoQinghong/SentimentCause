import re

def filter_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r";", " ;", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"[0-9]+[ -]*[0-9]*", ' __NUM__ ', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def get_text_position(text):
    text = text.split(u'\001')
    sentence = text[0]
    seq_len = len(sentence)
    words = sentence.split()
    aspect_term = text[1]
    sentence_position = []
    for word in words:
        from_position = sentence.index(word)
        to_position = from_position + len(word)-1
        sentence_position.append([from_position/seq_len,to_position/seq_len])
    term_from = sentence.index(aspect_term)
    term_to = term_from + len(aspect_term)-1
    return sentence_position,[term_from/seq_len,term_to/seq_len]
