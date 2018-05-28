#encoding:utf-8
import jieba
from collections import Counter
import numpy as np
import pickle
import codecs
from collections import defaultdict



def split(filetrainname,filetestname='../../Data/Milingual/emnlp_test.txt',language='C'):
    if language=='C':
        filetrainoriginalname = '../../Data/emnlp_train_nosampling.txt'
        filereadname = '../../Data/position_emnlp.txt'

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
    data_C = codecs.open('../../Data/position_emnlp.txt', 'r','utf-8').readlines()
    vec_new_C = codecs.open(wvC, 'r','utf-8').readlines()
 
    words_C = []
    words_C.extend(getWords(data_C))
 
    words_count_C = list(Counter(words_C).items())
    words_count_C = sorted(words_count_C, key=lambda x: -x[1])
    top_words_C, _ = zip(*words_count_C)

    top_words_C = ['__PAD__'] + list(top_words_C)
    word2id_C = dict(zip(top_words_C, range(len(top_words_C))))
    id2word_C = dict(zip(range(len(top_words_C)), top_words_C))


    embedding_C = np.random.normal(size=(len(word2id_C), 50), loc=0, scale=0.1)
    count = 0
    for line in vec_new_C:
        line = line.strip().split()
        if line[0] in word2id_C:
            embedding_C[word2id_C[line[0]]] = np.array([float(x) for x in line[1].split(',')])
            count += 1
    embedding_C[word2id_C['__PAD__']] = np.zeros(50)

    pickle.dump(embedding_C,open('../../Data/Milingual/emnlp_embedding.pkl','wb'))
def process():

    train_data_C = codecs.open('../../Data/emnlp_train_nosampling.txt', 'r','utf-8').readlines()
    test_data_C = codecs.open('../../Data/emnlp_test.txt', 'r','utf-8').readlines()
    
    words_C = []
    words_C.extend(getWords(train_data_C))
    words_C.extend(getWords(test_data_C))

    words_count_C = list(Counter(words_C).items())
    words_count_C = sorted(words_count_C, key=lambda x: -x[1])
    top_words_C, _ = zip(*words_count_C)

    top_words_C = ['__PAD__'] + list(top_words_C)
    word2id_C = dict(zip(top_words_C, range(len(top_words_C))))
    id2word_C = dict(zip(range(len(top_words_C)), top_words_C))

    position = []
    position.extend(get_position(train_data_C))
    position.extend(get_position(test_data_C))
    
    max_p = max(position)
    min_p = min(position)
    p_p = max_p + abs(min_p)

    train_zip_C_0 = []
    train_zip_C_1 = []
    test_zip_C = []
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
        if split_line[4]=='Y':
            train_zip_C_1.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
        else:
            
            train_zip_C_0.append([text_jieba, aspect_jieba, tmp, label, inx, clause_inx])
    print(len(train_zip_C_0),len(train_zip_C_1))
    times = int(len(train_zip_C_0)/len(train_zip_C_1))
    train_final_data_C = []
    batch_size = 32
    for time in range(times):
        temp = []
        for i in range(len(train_zip_C_1)):
            temp.append(train_zip_C_1[i])
            temp.append(train_zip_C_0[i+time*len(train_zip_C_1)])
        temp_sort = sorted(temp,key=lambda x:len(x[0]))
        bs=0
        while bs*batch_size<len(temp):
            tmp_data = temp[bs*batch_size:(bs+1)*batch_size]
            t, a, p, l, inx, clause_inx = zip(*tmp_data)
            batch_len = [len(x) for x in t]
            max_len = max(batch_len)
            pad_t = []
            for tmp_tt in t:
                tmp_tt = np.array(tmp_tt + [word2id_C['__PAD__']]*(max_len-len(tmp_tt)))
                pad_t.append(tmp_tt)
            train_final_data_C.append(list(zip(pad_t, list(a), p, l, inx, clause_inx)))
            bs += 1
       
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
        
    test_zip_sort_C = sorted(test_zip_C, key=lambda x: len(x[0]))  #20556 3484
    

    batch_size = 32
    i = 0
    test_final_data_C = []
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
    

    pickle.dump(train_final_data_C, open('../../Data/Milingual/emnlp_train.pkl', 'wb'))
    pickle.dump(test_final_data_C, open('../../Data/Milingual/emnlp_test.pkl', 'wb'))

if __name__=='__main__':
    #get_embedding()
    process()
