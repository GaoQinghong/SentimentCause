from collections import defaultdict
import xml.dom.minidom
import codecs
from  collections import Counter
import jieba

# 将XML格式文件转化为.txt格式文件
def read_xml(filereadname,filewritename):
    file_write = codecs.open(filewritename, 'w', 'utf-8')
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filereadname)
    all_sentences = DOMTree.documentElement
    sentences = all_sentences.getElementsByTagName("emotion")

    all = []
    for sentence in sentences:
        emotion_id = sentence.getAttribute("id")
        if sentence.hasAttribute("id"):
            print("id: %s" % sentence.getAttribute("id"))
        subsentences = sentence.getElementsByTagName('clause')
        emotion_category = sentence.getElementsByTagName('category')[0]
        emotion = emotion_category.getAttribute('name')
        print(emotion_id,len(subsentences))

        for sub in subsentences:
            try:
                clause_id = sub.getAttribute("id")
                cause = sub.getAttribute("cause")
                keyword = sub.getAttribute("keywords")
                text = sub.getElementsByTagName("text")[0].childNodes[0].data
                text = text.strip()
                if text=='':
                    continue
                print('text',text)
                if cause == "Y":
                    cause_text = sub.getElementsByTagName("cause")[0].childNodes[0].data
                    cause_id = sub.getElementsByTagName('cause')[0].attributes.getNamedItem('id').value

                else:
                    cause_text = "None"
                    cause_id = 0
                if keyword == "Y":
                    keywords_text = sub.getElementsByTagName("keywords")[0].childNodes[0].data
                else:
                    keywords_text = "None"
                all.append([str(emotion_id) ,str(clause_id) ,text , str(cause_id) ,str(cause) , cause_text ,str(keyword),keywords_text,emotion])
            except Exception as e:
                continue
    for item in all:
        file_write.writelines('\001'.join(item)+'\n')
    file_write.close()

#统计 原因子句&非原因子句 比例
def cause_proportion(filereadname):
    fileread = codecs.open(filereadname,'r','utf-8')
    cause_num = 0
    uncause_num = 0
    for line in fileread.readlines():
        line = line.strip().split('\001')
        if line[4] == 'Y':
            cause_num += 1
        else:
            uncause_num += 1

    print('cause : un cause ',cause_num,uncause_num)

    fileread.close()

def count_position(filereadname):
    position = []

    fileread = codecs.open(filereadname,'r','utf-8')
    for line in fileread.readlines():
        line = line.strip().split('\001')
        # print(line[-1])
        position.append(int(line[-1]))

    position_dict = sorted(Counter(position).items(),key=lambda x:x[1],reverse=True)
    print(position_dict)

    for key,value in position_dict:
        print(key,value/len(position))

def extract_generator_trainging_data(filereadname,filewritename):
    fileread = codecs.open(filereadname,'r','utf-8')
    filewrite = codecs.open(filewritename,'w','utf-8')
    sample_dict = defaultdict(list)
    for line in fileread.readlines():
        line = line.strip().split('\001')
        sample_dict[line[0]].append(line)
    count = 0
    for key,value in sample_dict.items():
        for v in value:
            filewrite.writelines('\001'.join(v)+'\n')
        count+=1
        if count==750:
            break
    filewrite.close()
    fileread.close()


def statistic_cause(filereadname):
    fileread = codecs.open(filereadname,'r','utf-8')
    cause_num_dict = defaultdict(int)
    for line in fileread.readlines():
        line = line.strip().split('\001')
        if line[4]=='Y':
            cause_num_dict[line[0]]+=1
    a = sorted(cause_num_dict.items(),key=lambda x:x[1])
    prob_dict = defaultdict(int)
    for value in a:
        prob_dict[value[1]] += 1
        print(value[0],value[1])
    prob = sorted(prob_dict.items(),key=lambda x:x[0])
    for value in prob:
        print(value[0],':',value[1]/len(cause_num_dict))
    fileread.close()

def get_emotion_expression(fileTrainReadname,fileTestReadname,filewritename):
    fileTrain = codecs.open(fileTrainReadname,'r','utf-8').readlines()
    fileTest = codecs.open(fileTestReadname,'r','utf-8').readlines()
    filewrite = codecs.open(filewritename,'w','utf-8')
    emotionC = []
    #emotionE = []
    for line in fileTrain:
        line = line.strip().split('\001')
        if line[6]=='Y':
            emotionC.append(line[7]+'\001'+line[8])
    for line in fileTest:
        line = line.strip().split('\001')
        if line[6]=='Y':
            emotionC.append(line[7]+'\001'+line[8])
    emotionC = set(emotionC)
    for item in emotionC:
        filewrite.writelines(item+'\n')
    filewrite.close()

def classifi_emo(filereadname1,filereadname2):
    fileread1 = codecs.open(filereadname1,'r','utf-8')
    fileread2 = codecs.open(filereadname2,'r','utf-8')
    # happiness = []
    # sadness = []
    # surprise = []
    # fear = []
    # anger = []
    # disgust = []
    emoE = {'happiness':[],'sadness':[],'surprise':[],'fear':[],'anger':[],'disgust':[]}
    emoC = {'happiness': [], 'sadness': [], 'surprise': [], 'fear': [], 'anger': [], 'disgust': []}
    for line in fileread1.readlines():
        line = line.strip().split('\001')
        emoE[line[1]].append(list(jieba.cut(line[0])))
    for line in fileread2.readlines():
        line = line.strip().split('\001')
        emoC[line[1]].append(list(jieba.cut(line[0])))
    print('------happiness-------')
    print(emoE['happiness'])
    print(emoC['happiness'])
    print('--------sadness--------')
    print(emoE['sadness'])
    print(emoC['sadness'])
    print('------surprise---------')
    print(emoE['surprise'])
    print(emoC['surprise'])
    print('-------fear--------')
    print(emoE['fear'])
    print(emoC['fear'])
    print('------anger--------')
    print(emoE['anger'])
    print(emoC['anger'])
    print('-----disgust-------')
    print(emoE['disgust'])
    print(emoC['disgust'])


if __name__ == "__main__":
    classifi_emo('../Data/emotionExpression/emotionE_C.txt','../Data/emotionExpression/emotionC.txt')
    # get_emotion_expression('../position_sampling/emnlp_train.txt','../position_sampling/emnlp_test.txt','../position_sampling/emotionC.txt')
    # get_emotion_expression('../position_sampling/emotion_cause_english_train_ns.txt','../position_sampling/emotion_cause_english_test.txt','../position_sampling/emotionE.txt')
    #read_xml('../NTCIR-ECA13-3000/emotion_cause_english_train.xml','../txt/emotion_cause_english_train_250.txt')

    #cause_proportion('../position_sampling/emnlp_train_sampling.txt')
    # cause_proportion('../txt/emotion_cause_english_test.txt')
    # cause_proportion('../txt/emotion_cause_english_train_ns.txt')
    #cause_proportion('../sampling_same_attention/position_english_test.txt')

    #count_position('../txt/emotion_cause_english_test.txt')
    #extract_generator_trainging_data('../position_sampling/emotion_cause_english_train_ns.txt','../generator_estimator/emotion_cause_english_train_750.txt')
    #statistic_cause('../txt/emotion_cause_english_test.txt')


    '''
    emnlp_train :
    0 0.5055614406779662
    -1 0.3302436440677966
    -2 0.055614406779661014
    1 0.0513771186440678
    -3 0.01694915254237288
    2 0.014830508474576272
    -4 0.007150423728813559
    -5 0.004502118644067797
    3 0.00423728813559322
    -6 0.0026483050847457626
    4 0.0015889830508474577
    -7 0.001059322033898305
    5 0.0007944915254237289
    6 0.0005296610169491525
    7 0.0005296610169491525
    8 0.0005296610169491525
    9 0.00026483050847457627
    10 0.00026483050847457627
    11 0.00026483050847457627
    12 0.00026483050847457627
    -10 0.00026483050847457627
    -9 0.00026483050847457627
    -8 0.00026483050847457627
    
    emnlp_test:
        
    0 0.49
    -1 0.3175
    1 0.06
    -2 0.0525
    2 0.0225
    3 0.01
    4 0.0075
    5 0.0075
    -3 0.0075
    6 0.005
    7 0.005
    -5 0.005
    -4 0.005
    8 0.0025
    -6 0.0025
    
    
    
    english train:
1 : 0.15378151260504203
2 : 0.7672268907563026
3 : 0.005042016806722689
4 : 0.058823529411764705
6 : 0.010084033613445379
8 : 0.003781512605042017
10 : 0.0008403361344537816
12 : 0.0004201680672268908

emnlp :
1 : 0.9719714964370546
2 : 0.02660332541567696
3 : 0.0014251781472684087
    '''
