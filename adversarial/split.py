#encoding:utf-8
import jieba
from collections import Counter
import numpy as np
import pickle
import codecs
from collections import defaultdict



def split(filetrainname,filetestname='../../Data/emnlp_test.txt',language='C'):
    if language=='C':
        filetrainoriginalname = '../../Data/emnlp_train_nosampling.txt'
        filereadname = '../../Data/position_emnlp.txt'
        ############emnlp##################
  
        fileread = codecs.open(filereadname, 'r', 'utf-8')
        filetrain = codecs.open(filetrainoriginalname, 'w', 'utf-8')
        filetest = codecs.open(filetestname, 'w', 'utf-8')

        samples = defaultdict(list)
        for line in fileread.readlines():
            line = line.strip()
            index = line.split('\001')[0]
            samples[index].append(line)
        print(len(samples))
        train_len = int(len(samples) * 0.9)
        count = 0
        for key, value in samples.items():
            if count < train_len:
                for v in value:
                    filetrain.writelines(v + '\n')
            else:
                for v in value:
                    filetest.writelines(v + '\n')
            count += 1
        fileread.close()
        filetrain.close()
        filetest.close()
        ####################################################
    if language=='C':
        filetrainoriginalname = '../../Data/emnlp_train_nosampling.txt'
    else:filetrainoriginalname = '../../Data/ntcir_train_ns.txt'

    fileread = codecs.open(filetrainoriginalname, 'r', 'utf-8')
    filewrite = codecs.open(filetrainname, 'w', 'utf-8')

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

  #  print(words)
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
    data_C = codecs.open('../../Data/position_emnlp.txt', 'r','utf-8').readlines()
    train_data_E = codecs.open('../../Data/ntcir_train.txt', 'r', 'utf-8').readlines()
    test_data_E = codecs.open('../../Data/ntcir_test.txt', 'r', 'utf-8').readlines()
    vec_new_C = codecs.open(wvC, 'r','utf-8').readlines()
    vec_new_E = codecs.open(wvE, 'r', 'utf-8').readlines()

    words_C = []
    words_C.extend(getWords(data_C))
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

    pickle.dump(embedding_C,open('../../Data/emnlp_embedding.pkl','wb'))
    pickle.dump(embedding_E, open('../../Data/ntcir_embedding.pkl','wb'))
def process():

    train_data_C = codecs.open('../../Data/emnlp_train.txt', 'r','utf-8').readlines()
    test_data_C = codecs.open('../../Data/emnlp_test.txt', 'r','utf-8').readlines()
    
    train_data_E = codecs.open('../../Data/ntcir_train.txt', 'r', 'utf-8').readlines()
    test_data_E = codecs.open('../../Data/ntcir_test.txt', 'r', 'utf-8').readlines()


    words_C = []
    words_C.extend(getWords(train_data_C))
    words_C.extend(getWords(test_data_C))
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
    position.extend(get_position(train_data_E))
    position.extend(get_position(test_data_E))

    max_p = max(position)
    min_p = min(position)
    p_p = max_p + abs(min_p)

    train_zip_C = []
    test_zip_C = []
    ###############################
    expanded_train_zip_C = {'happiness':[],'sadness':[],'fear':[],'disgust':[],'surprise':[],'anger':[]}
    ##############################
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
        #################################
        expanded_train_zip_C[split_line[8]].append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
        #################################
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
    
    train_zip_E = []
    test_zip_E = []
    ##################################
    expanded_train_zip_E = {'happiness': [], 'sadness': [], 'fear': [], 'disgust': [], 'surprise': [], 'anger': []}
    ################################
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
        #############################
        expanded_train_zip_E[split_line[8]].append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
        ##############################
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
    
    
    ###处理扩增的数据######
    #英文文本+中文情感   中文文本+英文情感
    train_final_data_EC = []
    batch_size = 32
    for key,value in expanded_train_zip_E.items():
            
        if len(expanded_train_zip_C[key])>len(value):
            expanded_train_zip_C[key] = expanded_train_zip_C[key][:len(value)]
        else: 
            times = int(len(value)/len(expanded_train_zip_C[key]))
            expanded_train_zip_C[key] = (expanded_train_zip_C[key]*(times+1))[:len(value)]   
        value = sorted(value,key=lambda x:len(x[0]))
        expanded_train_zip_E[key] = value

        i = 0
        while i * batch_size < len(value):
            tmp_data = value[i * batch_size:(i + 1) * batch_size]
            t, a, p, l, inx, clause_inx = zip(*tmp_data)
            batch_len = [len(x) for x in t]
            max_len = max(batch_len)
            pad_t = []
            for tmp_tt in t:
                tmp_tt = np.array(tmp_tt + [word2id_E['__PAD__']] * (max_len - len(tmp_tt)))
                pad_t.append(tmp_tt)
            train_final_data_EC.append(list(zip(pad_t, a, p, l, inx, clause_inx)))
            i += 1
    train_final_data_CE = []
    for key,value in expanded_train_zip_C.items():
        value = sorted(value,key=lambda x:len(x[0]))
        expanded_train_zip_C[key] = value
        i = 0
        while i * batch_size < len(value):
            tmp_data = value[i * batch_size:(i + 1) * batch_size]
            t, a, p, l, inx, clause_inx = zip(*tmp_data)
            batch_len = [len(x) for x in t]
            max_len = max(batch_len)
            pad_t = []
            for tmp_tt in t:
            
                tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']] * (max_len - len(tmp_tt)))
                pad_t.append(tmp_tt)
        
            train_final_data_CE.append(list(zip(pad_t, a, p, l, inx, clause_inx)))
            i += 1

    pickle.dump(train_final_data_EC,open('../../Data/ntcir_expand.pkl','wb'))
    pickle.dump(train_final_data_CE, open('../../Data/emnlp_expand.pkl', 'wb'))
    ######################

    train_zip_C = (train_zip_C*9)[:len(train_zip_E)]
    train_zip_sort_C = sorted(train_zip_C, key=lambda x: len(x[0])) # 3954 1695
    test_zip_sort_C = sorted(test_zip_C, key=lambda x: len(x[0]))  #20556 3484
    
    train_zip_sort_E = sorted(train_zip_E, key=lambda x: len(x[0]))
    test_zip_sort_E = sorted(test_zip_E, key=lambda x: len(x[0]))

    batch_size = 32
    i = 0
    train_final_data_C = []
    test_final_data_C = []
    train_final_data_E = []
    test_final_data_E = []
    while i*batch_size < len(train_zip_sort_E):
        tmp_data_E = train_zip_sort_E[i*batch_size:(i+1)*batch_size]
        tmp_data_C = train_zip_sort_C[i * batch_size:(i + 1) * batch_size]
        t_E, a_E, p_E, l_E, inx_E, clause_inx_E = zip(*tmp_data_E)
        t_C, a_C, p_C, l_C, inx_C, clause_inx_C = zip(*tmp_data_C)
        batch_len = [len(x) for x in t_C+t_E]
        max_len = max(batch_len)
        pad_t_E = []
        for tmp_tt in t_E:
            tmp_tt = np.array(tmp_tt + [word2id_E['__PAD__']]*(max_len-len(tmp_tt)))
            pad_t_E.append(tmp_tt)
        train_final_data_E.append(list(zip(pad_t_E, a_E, p_E, l_E, inx_E, clause_inx_E)))
        pad_t_C = []
        for tmp_tt in t_C:
            tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']] * (max_len - len(tmp_tt)))
            pad_t_C.append(tmp_tt)
        train_final_data_C.append(list(zip(pad_t_C, a_C, p_C, l_C, inx_C, clause_inx_C)))
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
#     print(test_zip_sort_C)
#     exit()
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
    

    pickle.dump(train_final_data_C, open('../../Data/emnlp_train.pkl', 'wb'))

    pickle.dump(test_final_data_C, open('../../Data/emnlp_test.pkl', 'wb'))
    
    pickle.dump(train_final_data_E, open('../../Data/ntcir_train.pkl', 'wb'))

    pickle.dump(test_final_data_E, open('../../Data/ntcir_test.pkl', 'wb'))


if __name__=='__main__':
    get_embedding()
    #process()
