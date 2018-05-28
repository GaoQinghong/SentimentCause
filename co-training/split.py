#encoding:utf-8
import jieba
from collections import Counter
import numpy as np
import pickle
import codecs
from collections import defaultdict


#对emnlp 中文数据集 以及 翻译成的EN_emnlp数据进行划分  测试集  及 最终验证集 
def split():
    Efileread = codecs.open('../../Data/co-training-data/EN_emnlp.txt','r','utf-8')
    Efiletrain = codecs.open('../../Data/co-training-data/EN_emnlp_train_ns.txt','w','utf-8')
    Efiletest = codecs.open('../../Data/co-training-data/EN_emnlp_test.txt','w','utf-8')
    
    Cfileread = codecs.open('../../Data/position_emnlp.txt','r','utf-8')
    Cfiletrain = codecs.open('../../Data/co-training-data/CH_emnlp_train_ns.txt','w','utf-8')
    Cfiletest = codecs.open('../../Data/co-training-data/CH_emnlp_test.txt','w','utf-8')
    
    train_index = []
    Esamples = defaultdict(list)
    Csamples = defaultdict(list)
    for line in Efileread.readlines():
        line = line.strip()
        index = line.split('\001')[0]
        Esamples[index].append(line)
    
    Etrain_len = int(len(Esamples) * 0.9)
    count = 0
    for key, value in Esamples.items():
        if count < Etrain_len:
            train_index.append(key)
            for v in value:
                Efiletrain.writelines(v + '\n')
        else:
            for v in value:
                Efiletest.writelines(v + '\n')
        count += 1
    for line in Cfileread.readlines():
        line = line.strip().split('\001')
        if line[0] in train_index:
            Cfiletrain.writelines(line+'\n')
        else:
            Cfiletest.writelines(line+'\n')
    Cfileread.close()
    Cfiletrain.close()
    Cfiletest.close()
    Efileread.close()
    Efiletrain.close()
    Efiletest.close()
       
#对中文训练集 以及 英文训练集 进行抽样   
def sampling(fileOriginalName,fileWriteName):
    fileread = codecs.open(fileOriginalName,'r','utf-8')
    filewrite = codecs.open(fileWriteName,'w','utf-8') 
 
    sentences_dict = defaultdict(list)
    sentences_interval = defaultdict(list)
    for line in fileread.readlines():
        line = line.strip().split('\001')
        length = len(sentences_dict[line[0]])
        line[1] = str(length + 1)
        sentences_dict[line[0]].append(line)
        if line[4] == 'Y' or line[6] == 'Y':
            sentences_interval[line[0]].append(int(line[1]))

    sentences_sampling = []
    for key, value in sentences_interval.items():
        sentences_sampling.extend(sentences_dict[key][value[0] - 1:value[-1]])

    for line in sentences_sampling:
        filewrite.writelines('\001'.join(line) + '\n')

    fileread.close()
    filewrite.close()
    
def getWords(data,language='C'):
    words = []
    for line in data:
        text = line.strip().split('\x01')[2].strip()
        aspect = line.strip().split('\x01')[7].strip()
        if language == 'C':
            words.extend(jieba.cut(text))
            words.extend(jieba.cut(aspect))
        else:
            words.extend(text.split())
            words.extend(aspect.split())
    return words

def get_position(data):
    position = []
    for line in data:
        p = line.strip().split('\x01')[-1].strip()
        position.append(int(p))
    return position

def get_embedding():

    wvC = '../../Data/biling_trained_Zh_vectors.txt'
    wvE = '../../Data/En_vectors.txt'
    dataC1 = codecs.open('../../Data/co-training-data/CH_ntcir_train_ns.txt', 'r','utf-8').readlines()
    dataC2 = codecs.open('../../Data/position_emnlp.txt','r','utf-8').readlines()
    dataE1 = codecs.open('../../Data/ntcir_train_ns.txt', 'r', 'utf-8').readlines()
    dataE2 = codecs.open('../../Data/co-training-data/EN_emnlp.txt', 'r', 'utf-8').readlines()
    vec_new_C = codecs.open(wvC, 'r','utf-8').readlines()
    vec_new_E = codecs.open(wvE, 'r', 'utf-8').readlines()

    words_C = []
    words_C.extend(getWords(dataC1))
    words_C.extend(getWords(dataC2))
    words_E = []
    words_E.extend(getWords(dataE1,language='E'))
    words_E.extend(getWords(dataE2,language='E'))

    words_count_C = list(Counter(words_C).items())
    words_count_C = sorted(words_count_C, key=lambda x: -x[1])
    top_words_C, _ = zip(*words_count_C)

    top_words_C = ['__PAD__'] + list(top_words_C)
    word2id_C = dict(zip(top_words_C, range(len(top_words_C))))
    id2word_C = dict(zip(range(len(top_words_C)), top_words_C))

    words_count_E = list(Counter(words_E).items())
    words_count_E = sorted(words_count_E, key=lambda x: -x[1])
    top_words_E, _ = zip(*words_count_E)

    top_words_E = ['__PAD__'] + list(top_words_E)
    word2id_E = dict(zip(top_words_E, range(len(top_words_E))))
    id2word_E = dict(zip(range(len(top_words_E)), top_words_E))

    embedding_C = np.random.normal(size=(len(word2id_C), 50), loc=0, scale=0.1)
    count = 0
    for line in vec_new_C:
        line = line.strip().split()
        if line[0] in word2id_C:
            embedding_C[word2id_C[line[0]]] = np.array([float(x) for x in line[1].split(',')])
            count += 1
    embedding_C[word2id_C['__PAD__']] = np.zeros(50)

    embedding_E = np.random.normal(size=(len(word2id_E), 50), loc=0, scale=0.1)
    count = 0
    for line in vec_new_E:
        line = line.strip().split()
        if line[0] in word2id_E:
            embedding_E[word2id_E[line[0]]] = np.array([float(x) for x in line[1].split(',')])
            count += 1
    embedding_E[word2id_E['__PAD__']] = np.zeros(50)

    pickle.dump(embedding_C,open('../../Data/co-training-data/chinses_embedding.pkl','wb'))
    pickle.dump(embedding_E, open('../../Data/co-training-data/english_embedding.pkl','wb'))
                
def process():
    
    train_data_C = codecs.open('../../Data/co-training-data/CH_ntcir_train.txt', 'r','utf-8').readlines()  #中文训练数据
    test_data_C = codecs.open('../../Data/co-training-data/CH_emnlp_train.txt', 'r','utf-8').readlines()   #中文无标签数据
    valid_data_C = codecs.open('../../Data/co-training-data/CH_emnlp_test.txt', 'r','utf-8').readlines()   #最终验证数据
    
    train_data_E = codecs.open('../../Data/co-training-data/EN_ntcir_train.txt', 'r', 'utf-8').readlines()    #英文训练数据
    test_data_E = codecs.open('../../Data/co-training-data/EN_emnlp_train.txt', 'r', 'utf-8').readlines()   #英文无标签数据

    words_C = []
    words_C.extend(getWords(train_data_C))
    words_C.extend(getWords(test_data_C))
    words_C.extend(getWords(valid_data_C))
    words_E = []
    words_E.extend(getWords(train_data_E,language='E'))
    words_E.extend(getWords(test_data_E,language='E'))

    words_count_C = list(Counter(words_C).items())
    words_count_C = sorted(words_count_C, key=lambda x: -x[1])
    top_words_C, _ = zip(*words_count_C)

    top_words_C = ['__PAD__'] + list(top_words_C)
    word2id_C = dict(zip(top_words_C, range(len(top_words_C))))
    id2word_C = dict(zip(range(len(top_words_C)), top_words_C))

    words_count_E = list(Counter(words_E).items())
    words_count_E = sorted(words_count_E, key=lambda x: -x[1])
    top_words_E, _ = zip(*words_count_E)

    top_words_E = ['__PAD__'] + list(top_words_E)
    word2id_E = dict(zip(top_words_E, range(len(top_words_E))))
    id2word_E = dict(zip(range(len(top_words_E)), top_words_E))

    position = []
    position.extend(get_position(train_data_C))
    position.extend(get_position(test_data_C))
    position.extend(get_position(valid_data_C))
    position.extend(get_position(train_data_E))
    position.extend(get_position(test_data_E))

    max_p = max(position)
    min_p = min(position)
    p_p = max_p + abs(min_p)

    train_zip_C = []
    test_zip_C = []
    valid_zip_C = []
   
    for line in train_data_C:
        split_line = line.strip().split('\001')
        inx = split_line[0]
        clause_inx = split_line[1]
        text = split_line[2].strip()
        label = np.array([0, 1]) if split_line[4] == 'Y' else np.array([1, 0])
        aspect = split_line[7].strip()
        position = int(split_line[-1].strip())
        text_jieba = [word2id_C[x] for x in jieba.cut(text)]
        aspect_jieba = [[word2id_C[x] for x in jieba.cut(aspect)][0]]
        tmp = np.zeros(p_p+1)
        tmp[position+abs(min_p)] = 1.
        train_zip_C.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
      
    for line in test_data_C:
        split_line = line.strip().split('\x01')
        inx = split_line[0]
        clause_inx = split_line[1]
        text = split_line[2].strip()
        label = np.array([0, 1]) if split_line[4] == 'Y' else np.array([1, 0])
        aspect = split_line[7].strip()
        position = int(split_line[-1].strip())
        text_jieba = [ word2id_C[x] for x in jieba.cut(text)]
        aspect_jieba = [[word2id_C[x] for x in jieba.cut(aspect)][0]]
        tmp = np.zeros(p_p+1)
        tmp[position+abs(min_p)] = 1.
        test_zip_C.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
        
    for line in valid_data_C:
        split_line = line.strip().split('\x01')
        inx = split_line[0]
        clause_inx = split_line[1]
        text = split_line[2].strip()
        label = np.array([0, 1]) if split_line[4] == 'Y' else np.array([1, 0])
        aspect = split_line[7].strip()
        position = int(split_line[-1].strip())
        text_jieba = [ word2id_C[x] for x in jieba.cut(text)]
        aspect_jieba = [[word2id_C[x] for x in jieba.cut(aspect)][0]]
        tmp = np.zeros(p_p+1)
        tmp[position+abs(min_p)] = 1.
        valid_zip_C.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
    
    train_zip_E = []
    test_zip_E = []
   
    for line in train_data_E:
        split_line = line.strip().split('\x01')
        inx = split_line[0]
        clause_inx = split_line[1]
        text = split_line[2].strip()
        label = np.array([0, 1]) if split_line[4] == 'Y' else np.array([1, 0])
        aspect = split_line[7].strip().split()
        if len(aspect)>=5:
            aspect = aspect[:5]
        else: aspect = aspect+['__PAD__']*(5-len(aspect))
        if len(aspect)!=5:
            print(len(aspect))
        position = int(split_line[-1].strip())
        text_jieba = [word2id_E[x] for x in text.split()]
        aspect_jieba = [word2id_E[x] for x in aspect]
        tmp = np.zeros(p_p + 1)
        tmp[position + abs(min_p)] = 1.
        train_zip_E.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
       
    for line in test_data_E:
        split_line = line.strip().split('\x01')
        inx = split_line[0]
        clause_inx = split_line[1]
        text = split_line[2].strip()
        label = np.array([0, 1]) if split_line[4] == 'Y' else np.array([1, 0])
        aspect = split_line[7].strip().split()
        if len(aspect)>=5:
            aspect = aspect[:5]
        else: aspect = aspect+['__PAD__']*(5-len(aspect))
        if len(aspect)!=5:
            print(len(aspect))
        position = int(split_line[-1].strip())
        text_jieba = [word2id_E[x] for x in text.split()]
        aspect_jieba = [word2id_E[x] for x in aspect]
        tmp = np.zeros(p_p + 1)
        tmp[position + abs(min_p)] = 1.
        test_zip_E.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
        

    train_zip_C = (train_zip_C*9)[:len(train_zip_E)]
    train_zip_sort_C = sorted(train_zip_C, key=lambda x: len(x[0])) # 3954 1695
    test_zip_sort_C = sorted(test_zip_C, key=lambda x: len(x[0]))  #20556 3484
    valid_zip_sort_C = sorted(valid_zip_C,key=lambda x: len(x[0]))
    
    train_zip_sort_E = sorted(train_zip_E, key=lambda x: len(x[0]))
    test_zip_sort_E = sorted(test_zip_E, key=lambda x: len(x[0]))

    i = 0
    train_final_data_C = []
    test_final_data_C = []
    valid_final_data_C = []
    train_final_data_E = []
    test_final_data_E = []

    while i*batch_size < len(train_zip_sort_C):  
        tmp_data = train_zip_sort_C[i*batch_size:(i+1)*batch_size]
        t, a, p, l, inx, clause_inx = zip(*tmp_data)
        batch_len = [len(x) for x in t]
        max_len = max(batch_len)
        pad_t = []
        for tmp_tt in t:
            tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t.append(tmp_tt)
        train_final_data_C.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
        i += 1
    i = 0
    while i*batch_size < len(test_zip_sort_C):  
        tmp_data = test_zip_sort_C[i*batch_size:(i+1)*batch_size]
        t, a, p, l, inx, clause_inx = zip(*tmp_data)
        batch_len = [len(x) for x in t]
        max_len = max(batch_len)
        pad_t = []
        for tmp_tt in t:
            tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t.append(tmp_tt)
        test_final_data_C.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
        i += 1
    i = 0
    while i*batch_size < len(valid_zip_sort_C):  
        tmp_data = valid_zip_sort_C[i*batch_size:(i+1)*batch_size]
        t, a, p, l, inx, clause_inx = zip(*tmp_data)
        batch_len = [len(x) for x in t]
        max_len = max(batch_len)
        pad_t = []
        for tmp_tt in t:
            tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t.append(tmp_tt)
        valid_final_data_C.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
        i += 1
    i = 0
    while i*batch_size < len(train_zip_sort_E):
        tmp_data = train_zip_sort_E[i*batch_size:(i+1)*batch_size]
        t, a, p, l, inx, clause_inx = zip(*tmp_data)
        batch_len = [len(x) for x in t]
        max_len = max(batch_len)
        pad_t = []
        for tmp_tt in t:
            tmp_tt = np.array(tmp_tt + [word2id_E['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t.append(tmp_tt)
        train_final_data_E.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
        i += 1
    i = 0
    while i*batch_size < len(test_zip_sort_E):
        tmp_data = test_zip_sort_E[i*batch_size:(i+1)*batch_size]
        t, a, p, l, inx, clause_inx = zip(*tmp_data)
        batch_len = [len(x) for x in t]
        max_len = max(batch_len)
        pad_t = []
        for tmp_tt in t:
            tmp_tt = np.array(tmp_tt + [word2id_E['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t.append(tmp_tt)
        test_final_data_E.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
        i += 1
     
    pickle.dump(train_final_data_C, open('../../Data/co-training-data/CH_ntcir_train.pkl', 'wb')) #中文训练数据
    pickle.dump(test_final_data_C, open('../../Data/co-training-data/CH_emnlp_train.pkl', 'wb')) #中文无标签数据
    pickle.dump(valid_final_data_C, open('../../Data/co-training-data/CH_emnlp_test.pkl', 'wb')) #最终验证数据
    pickle.dump(train_final_data_E, open('../../Data/co-training-data/EN_ntcir_train.pkl', 'wb'))     #英文训练数据
    pickle.dump(test_final_data_E, open('../../Data/co-training-data/EN_emnlp_train.pkl', 'wb'))   #英文无标签数据


if __name__=='__main__':
    get_embedding()
    #process()
