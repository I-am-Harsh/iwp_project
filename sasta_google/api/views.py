from django.http import JsonResponse
import json
import pprint
import math
from nltk.stem import PorterStemmer
import re
from bs4 import BeautifulSoup
import requests

ps = PorterStemmer()
global_index = {}
global_directory = []
global_vocabulary = []
global_sizes = []
global_autocomplete = []

import json
from os import system, name
# import PyPDF2, re
import pprint
from os import listdir
from os.path import isfile, join
# import textract
from nltk.stem import PorterStemmer

ps = PorterStemmer()
global_directory = []
global_vocabulary = []
global_sizes = []
autocomplete = {}
# text = page_content.encode('utf-8')
# text = text.replace('\n','')

# 175,300
# 225,320

# text_split_1 = textract.process('./data/15147_split_1.pdf')
# text_split_2 = textract.process('./data/15147_split_2.pdf')
# text_split_3 = textract.process('./data/15147_split_3.pdf')

def buildIndexes(filenames):
    def clean(text):
        text = text.replace("\n\n", "{super-special-marker}")
        text = text.replace("\n", "")
        text = text.replace("{super-special-marker}", "\n")
        return text


    def get_text(file_location):
        # pdf_file = open(file_location, 'rb')
        # read_pdf = PyPDF2.PdfFileReader(pdf_file)
        content = ""
        # text = textract.process(file_location)
        text="lmfao"

        # for i in range(read_pdf.getNumPages()):
        #     page = read_pdf.getPage(i)
        #     content += clean(page.extractText())

        return text.lower()

    def process_files(filenames):

        file_to_terms = {}
        file_counter = 0
        for file in filenames:
            global_directory.append(file)
            pattern = re.compile('[\W_]+')
            file_to_terms[file_counter] = get_text(file)
            file_to_terms[file_counter] = ''.join(e if e.isalnum() else " " for e in file_to_terms[file_counter])
            re.sub(r'[\W_]+','', file_to_terms[file_counter])
            file_to_terms[file_counter] = file_to_terms[file_counter].split()
            for i, term in enumerate(file_to_terms[file_counter]):
                file_to_terms[file_counter][i] = ps.stem(term)
            global_sizes.append( len(file_to_terms[file_counter]) )
            file_counter += 1

        return file_to_terms

    def index_one_file(termlist):
        fileIndex = {}
        for index, word in enumerate(termlist):
            if word in fileIndex.keys():
                fileIndex[word].append(index)
            else:
                fileIndex[word] = [index]
        return fileIndex

    def make_indices(termlists):
        total = {}
        for filename in termlists.keys():
            total[filename] = index_one_file(termlists[filename])
        return total

    def fullIndex(regdex):
        total_index = {}
        for filename in regdex.keys():
            for word in regdex[filename].keys():

                if word in total_index.keys():
                    if filename in total_index[word].keys():
                        total_index[word][filename].extend(regdex[filename][word][:])
                    else:
                        total_index[word][filename] = regdex[filename][word]
                else:
                    total_index[word] = {filename: regdex[filename][word]}
        return total_index

    def one_word_query(word, invertedIndex):
        pattern = re.compile('[\W_]+')
        word = pattern.sub(' ',word)
        if word in invertedIndex.keys():
            return [filename for filename in invertedIndex[word].values()]
        else:
            return []

    def free_text_query(string,index):
        pattern = re.compile('[\W_]+')
        string = pattern.sub(' ',string)
        result = []
        for word in string.split():
            result += one_word_query(word,index)
        # print(result)
        return list(set(result))

    def phrase_query(string, invertedIndex):
        pattern = re.compile('[\W_]+')
        string = pattern.sub(' ',string)
        listOfLists, result = [],[]
        for word in string.split():
            listOfLists.append(free_text_query(word,invertedIndex))
        setted = set(listOfLists[0]).intersection(*listOfLists)
        for filename in setted:
            temp = []
            for word in string.split():
                temp.append(invertedIndex[word][filename][:])
            for i in range(len(temp)):
                for ind in range(len(temp[i])):
                    temp[i][ind] -= i
            if set(temp[0]).intersection(*temp):
                result.append(filename)
            # print('\n temp : \n')
            # print(temp)
        return result

    # filenames=['./data/15147_split_1.pdf']
    # # print(filenames)
    termslist=process_files(filenames)
    print(termslist)
    totaldict=make_indices(termslist)
    pprint.pprint(totaldict.keys())
    count=0

    index=fullIndex(totaldict)
    # # print('full index \n')
    print(index)
    # # print('\n\n')
    # # print(global_directory)

    word_count = 0
    smaller_index = {}
    for x in index:
        global_vocabulary.append(x)
        # # print(index[x])
        smaller_index[int(word_count)] = index[x]
        word_count += 1

    for i in smaller_index:
        # # print( global_vocabulary[i] )
        autocomplete[ global_vocabulary[i] ] = sum(len(v) for v in smaller_index[i].values())

    autocomplete_list_ = sorted(autocomplete.items(), key=lambda x: autocomplete.get(x[0]))
    autocomplete_list = []

    for i in autocomplete_list_:
        autocomplete_list.append( i[0] )

    # with open("autocomplete.json","w") as f:
    for i in list(reversed(autocomplete_list)):
        # f.write( str(i) + "\n" )
        global_autocomplete.append(str(i))

    # with open("index.json", "w") as f:
    #     json.dump(smaller_index, f)

    global_index = smaller_index
    # print(global_index)
    # with open("directory.json", "w") as f:
    # for i in global_directory:
    #     # f.write(str(i) + "\n")
    #     global_directory.append(str(i))

    # with open("vocabulary.json", "w") as f:
    # for i in global_vocabulary:
    #     # f.write(str(i) + "\n")
    #     global_vocabulary.append(str(i))

    print(global_vocabulary)
    # with open("global_sizes.json", "w") as f:
    # for i in global_sizes:
    #     global_sizes.append(str(i))
            # f.write(str(i) + "\n")
    # print("as")
    return(global_index)
# # print(smaller_index)
# # print(global_vocabulary)
# search_term = input("Enter Term to Search for\n")
# # # print(phrase_query(search_term,index))
# try: # print(index[search_term])
# except: # print("Not in files")

# LMFAO

# prefix = "../../"
prefix = ""

# with open( prefix + "index.json", "r") as f:
#     global_index = json.load(f)


# with open(prefix + "autocomplete.json", "r") as f:
#     for line in f:
#         global_autocomplete.append(line.strip())

# with open(prefix + "global_sizes.json", "r") as f:
#     for line in f:
#         global_sizes.append(line.strip())

# with open(prefix + "directory.json", "r") as f:
#     for line in f:
#         global_directory.append(line.strip())

# with open(prefix + "vocabulary.json", "r") as f:
#     for line in f:
#         global_vocabulary.append(line.strip())

def getFinalSearch(search_term,global_index):
    print(global_index)
    old_index = global_index
    global_index = {int(k):{int(i):[int(j) for j in global_index[k][i] ] for i in v} for k,v in global_index.items()}

    total_docs = len(global_directory) + 1

    def get_total_frequency(filename):
        return( global_sizes[ global_directory.index(filename) ] )

    def search(search_term):

        pattern = re.compile('[\W_]+')

        # search_terms = str(search_term).split("")
        search_terms = ''.join(e if e.isalnum() else " " for e in search_term)
        re.sub(r'[\W_]+','', search_terms)

        search_terms = search_terms.split()

        for i, term in enumerate( search_terms ):
            search_terms[i] = ps.stem(term)
        total_occurences = {}
        print(search_terms)

        count = 0
        total_search_index = {}

        found_words = {}
        #Building Iniital Index with only documents containing search terms
        for i in search_terms:
            try:
                term_index = global_index [ global_vocabulary.index(i) ]
                for j in term_index:
                    document_location = global_directory[j]
                    if(document_location in total_search_index):
                        temp_list = total_search_index[ document_location ]
                        if( j in temp_list ):
                            temp_list_inner = temp_list[j]
                            temp_list_inner.append( global_index[global_vocabulary.index(i)][j] )
                            total_search_index[ document_location ] = temp_list_inner
                        else:
                            total_search_index[document_location][i] = global_index[global_vocabulary.index(i)][j]

                    else:
                        total_search_index[ document_location ] = { i : global_index[global_vocabulary.index(i)][j] }
            except:
                print("nahi he bro")

        score = {}
        idf = {}

        # print(total_search_index)
        for i in total_search_index:
            for j in total_search_index[i].keys():
                if i in found_words:
                    temp_list = found_words[i]
                    temp_list.append( j )
                    found_words[i] = temp_list
                else:
                    found_words[i] =  [j]
        # print(found_words)

        # for i in search_terms:
        #     try:
        #         idf[i] =
        #     except:
        #         print("Nahi mila")

        #For each term closenss
        for i in total_search_index:

            for j in search_terms:
                term_frequency = total_search_index[i]
                try:
                    if( i in score ):
                        score[i] += math.log( total_docs / len(old_index[str(global_vocabulary.index(j))].keys()) ) * (len(term_frequency[j])/int(get_total_frequency(i)))
                    else:
                        score[i] = math.log( total_docs / len(old_index[str(global_vocabulary.index(j))].keys()) ) * (len(term_frequency[j])/int(get_total_frequency(i)))
                except:
                    print('',end='')


            if( len(total_search_index[i]) > 1 ):
                keys = list(total_search_index[i].keys() )

                for x in range(len(keys)):
                    for z in total_search_index[i][keys[x]] :
                        # print(total_search_index[i][keys[x]])
                        xx = x+1
                        if(xx<len(keys)):
                            for aa in total_search_index[i][keys[xx]]:
                                if( aa - z < 7 and aa - z > 0 ):
                                    print(i,keys[xx],keys[x],aa,z)
                                    # score[i] += math.exp(z-aa)*100
                                    score[i] += 10*math.log( total_docs / len(old_index[str(global_vocabulary.index(j))].keys()) ) * math.exp(z-aa)

                                    print(math.exp(z-aa)*100)



                    # for z in total_search_index[i][j]:
                    #     print(z)

        #For sequence finding
        print(score)
        responses = {}
        responses['data'] = sorted( score.items(), key=lambda x: (len(found_words[x[0]]), score.get(x[0])))
        responses['found_words'] = found_words
        responses['score'] = score
        # responses['context'] = context

        return(responses)

    return(search(search_term))

def autoComplete():
    return(global_autocomplete)

def getSearch(request):
    body_unicode = request.body.decode('utf-8')
    req = body_unicode
    query = str(req).split('=')[1]
    return JsonResponse(search(query), status=200, safe=False )

def postURL(request):
    url = request.POST.get('url')
    print(url.split(','))
    global_index = buildIndexes( url.split(',') )
    print('lmfao')
    return JsonResponse( getFinalSearch( request.POST.get('keywords').split(','), global_index ) , status=200, safe=False )

def getAutocomplete(request):
    return JsonResponse( autoComplete(), status=200, safe=False )

def getSentiment():

    import numpy as np
    import matplotlib.pyplot as pp
    from random import  shuffle


    val = 0. # this is the value where you want the data to appear on the y-axis.

    pos_tweets = []
    pos_tweets_id = []
    neg_tweets = []
    neg_tweets_id = []


    pos_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentPosScore.txt')
    neg_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentNegScore.txt')

    read_pos = pos_tweets_file.read().split(",")
    for row in read_pos:
        pos_tweets_id.append(int(row.split(":")[0]))
        pos_tweets.append(float(row.split(":")[1]))

    read_neg = neg_tweets_file.read().split(",")
    for row in read_neg:
        neg_tweets_id.append(int(row.split(":")[0]))
        neg_tweets.append(float(row.split(":")[1]))

    pos_tweets = np.array(pos_tweets)
    neg_tweets = np.array(neg_tweets)

    shuffle(pos_tweets)
    shuffle(neg_tweets)

    ## pos_vs_neg_plot
    #pp.plot(pos_tweets, 'r--', neg_tweets, 'g--')

    ## cluster with centroids plots for our code
    #senti_read = open('coding_output.txt')

    ## cluster with centroids plots for skfuzzy
    #senti_read = open('fuzzy_output.txt')

    ## cluster with centroids plots for kmeans
    senti_read = open('/home/harsh/Downloads/sasta_google/api/kmeans_output.txt')


    line = senti_read.readlines()[0]
    line = line[1:len(line)-2]
    centroids = []
    for word in line.split():
        centroids.append(float(word))


    max_len = len(pos_tweets)
    if(len(pos_tweets) < len(neg_tweets)):
        max_len = len(neg_tweets)

    pp.scatter(pos_tweets,  np.zeros_like(pos_tweets) + val, c='red')
    pp.scatter(neg_tweets,  np.zeros_like(neg_tweets) + val, c='green')
    pp.scatter(centroids,  np.zeros_like(centroids) + val, c='black')

    #pp.scatter(max_len, neg_tweets, c='green')
    return(pp.show())            



##########    Graph
def sentimentView(request):
    return(JsonResponse( getSentiment(),status=200, safe=False ))





###########################  K means algorithm
def getkMeans():


        #!/usr/bin/python
        # -*- coding: UTF-8 -*-

        #----- Use python 2.7-------


        import skfuzzy as fuzz
        import matplotlib.pyplot as plt

        import sys
        # reload(sys)
        # sys.setdefaultencoding('utf-8')

        import nltk
        from emoji import UNICODE_EMOJI
        from nltk.stem.snowball import SnowballStemmer
        import re
        import numpy as np
        np.set_printoptions(threshold=np.nan)

        #from  sklearn_extensions.kmeans import FuzzyKMeans
        from sklearn.cluster import KMeans
        #time for execution
        import timeit
        start_time = timeit.default_timer()


        from random import  shuffle
        import math

        targetEmoticons = {1: "happy", 2: "love", 3: "playful", 4: "sad", 5: "angry", 6: "confused"}


        ## get positive and negative tweets from sentimentPosScore.txt
        ## and sentimentNegScore.txt which contains sentiment scores for positive and negative tweets

        pos_tweets = []
        pos_tweets_id = []
        neg_tweets = []
        neg_tweets_id = []

        pos_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentPosScore.txt')
        neg_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentNegScore.txt')

        read_pos = pos_tweets_file.read().split(",")
        for row in read_pos:
            pos_tweets_id.append(int(row.split(":")[0]))
            pos_tweets.append(float(row.split(":")[1]))

        read_neg = neg_tweets_file.read().split(",")
        for row in read_neg:
            neg_tweets_id.append(int(row.split(":")[0]))
            neg_tweets.append(float(row.split(":")[1]))

        pos_tweets = np.array(pos_tweets)
        neg_tweets = np.array(neg_tweets)

        shuffle(pos_tweets)
        shuffle(neg_tweets)

        ##add positive and negative scores to sentimentScore array and shuffle the array to avoid ordered data
        sentimentScore = []

        sentimentScore.extend(pos_tweets.flatten())
        sentimentScore.extend(neg_tweets.flatten())

        shuffle(sentimentScore)
        sentimentScore = np.array(sentimentScore)
        alldata = sentimentScore.reshape(-1, 1)


        #### KMeans
        #print('kmeans')
        #print ("---------")

        kmeans = KMeans(n_clusters=6)
        kmeans.fit(alldata)

        print (np.sort(np.array(kmeans.cluster_centers_).flatten()))
        # print(kmeans.cluster_centers_)
        # print (kmeans.labels_)
        # print (kmeans.cluster_centers_)
        # print (kmeans.n_clusters)

        #contains predicted centroids for all 6 emoticons, will be printed in kmeans_output.txt
        my_cluster_centers = np.array(kmeans.cluster_centers_).flatten().tolist()
        my_cluster_centers = sorted(my_cluster_centers)
        negative_centers = np.array(my_cluster_centers[:3]).flatten()
        positive_centers = np.array(my_cluster_centers[3:]).flatten()
        #print positive_centers, negative_centers


        ## cluster all data into 6 groups depending on which centroid its closest to
        ## e.g. 0.21 will be closest to centroid 0.22 as opposed to 0.08
        finalClusters = []

        def getClusters(sentiCenter, senti_tweets):
            cluster1 = []
            cluster2 = []
            cluster3 = []

            for i in range(len(senti_tweets)):

                dist1 = math.fabs(sentiCenter[0] - senti_tweets[i])
                dist2 = math.fabs(sentiCenter[1] - senti_tweets[i])
                dist3 = math.fabs(sentiCenter[2] - senti_tweets[i])

                if (dist1 > dist2):
                    if (dist2 > dist3):
                        cluster3.append(senti_tweets[i])
                    else:
                        cluster2.append(senti_tweets[i])
                else:
                    if (dist1 > dist3):
                        cluster3.append(senti_tweets[i])
                    else:
                        cluster1.append(senti_tweets[i])

            finalClusters.append(cluster1)
            finalClusters.append(cluster2)
            finalClusters.append(cluster3)


        ### get Clusters for positive tweets
        finalClustersDictIdx = 1
        getClusters(positive_centers, pos_tweets)

        ### get Clusters for negative tweets
        finalClustersDictIdx = 4
        getClusters(negative_centers, neg_tweets)


        ### combine positive and negative clusters and assign index of the predicted value to finalClustersIdx
        ### This will be the predicted target
        finalClustersDict = {1:"", 2:"", 3:"", 4:"", 5:"", 6:""}
        finalClustersIdx = []

        k = 0
        for clusters in finalClusters:
            finalClustersIdx.append([])

            for cluster in clusters:
                myidx = my_cluster_centers.index(min(my_cluster_centers, key=lambda x: abs(x - cluster)))+1
                #print cluster, my_cluster_centers,  myidx
                finalClustersIdx[k].append(myidx)
            k += 1

        ### get max value in each cluster. This will be the actual target
        indices = []
        for cluster in finalClustersIdx:
            try:
                indices.append(max(cluster,key=cluster.count))
            except:
                pass

        #print indices


        ####getting final accuracy from predicted target(each value in cluster array) vs actual target(indices)
        accuracy = [0]*len(my_cluster_centers)

        k = 0
        for cluster in finalClustersIdx:
            for row in cluster:
                try:
                    if(row == indices[k]):
                        accuracy[k] += 1
                except:
                    pass
            k += 1

        for row in range(len(accuracy)):
            #print  accuracy[row], len((finalClusters[row]))

            try:
                accuracy[row] = accuracy[row]/float(len(finalClusters[row]))*100
            except:
                pass

        #print "Accuracy for individual emotions", accuracy
        return(print ("Average accuracy", sum(accuracy)/len(accuracy)))


def kMeans(request):
    return(JsonResponse( getkMeans(),status=200, safe=False ))


def getfuzzyCmeans():

    #!/usr/bin/python
    # -*- coding: UTF-8 -*-

    #----- Use python 2.7-------


    import skfuzzy as fuzz
    import matplotlib.pyplot as plt

    import sys
    # reload(sys)
    # sys.setdefaultencoding('utf-8')

    import nltk
    from emoji import UNICODE_EMOJI
    from nltk.stem.snowball import SnowballStemmer
    import re
    import numpy as np
    np.set_printoptions(threshold=np.nan)

    from  sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

    #time for execution
    import timeit
    start_time = timeit.default_timer()


    from random import  shuffle
    import math

    targetEmoticons = {1: "happy", 2: "love", 3: "playful", 4: "sad", 5: "angry", 6: "confused"}


    ## get positive and negative tweets from sentimentPosScore.txt
    ## and sentimentNegScore.txt which contains sentiment scores for positive and negative tweets
    pos_tweets = []
    pos_tweets_id = []
    neg_tweets = []
    neg_tweets_id = []

    pos_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentPosScore.txt')
    neg_tweets_file = open('/home/harsh/Downloads/sasta_google/api/sentimentNegScore.txt')

    read_pos = pos_tweets_file.read().split(",")
    for row in read_pos:
        pos_tweets_id.append(int(row.split(":")[0]))
        pos_tweets.append(float(row.split(":")[1]))

    read_neg = neg_tweets_file.read().split(",")
    for row in read_neg:
        neg_tweets_id.append(int(row.split(":")[0]))
        neg_tweets.append(float(row.split(":")[1]))

    pos_tweets = np.array(pos_tweets)
    neg_tweets = np.array(neg_tweets)

    shuffle(pos_tweets)
    shuffle(neg_tweets)


    ### add positive and negative scores to sentimentScore array and shuffle the array to avoid ordered data
    sentimentScore = []

    sentimentScore.extend(pos_tweets.flatten())
    sentimentScore.extend(neg_tweets.flatten())
    shuffle(sentimentScore)
    sentimentScore = np.array(sentimentScore)
    alldata = sentimentScore.reshape(-1, 1)


    #### We are using FuzzyKMeans from skfuzzy because fuzzyCMeans of skfuzzy needs 2 dimensional data
    #### and we have one-dimensional data.
    ####http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

    #### Fuzzy KMeans
    #print('FUZZY_KMEANS')

    fuzzy_kmeans = FuzzyKMeans(k=6, m=2)
    fuzzy_kmeans.fit(alldata)
    print (np.sort((np.array(fuzzy_kmeans.cluster_centers_).flatten())))
    # print(fuzzy_kmeans.cluster_centers_)
    # print (kmeans.labels_)
    # print (kmeans.cluster_centers_)
    # print (kmeans.n_clusters)


    ### contains predicted centroids for all 6 emoticons, will be printed in fuzzy_output.txt
    my_cluster_centers = np.array(fuzzy_kmeans.cluster_centers_).flatten().tolist()
    my_cluster_centers = sorted(my_cluster_centers)
    negative_centers = np.array(my_cluster_centers[:3]).flatten()
    positive_centers = np.array(my_cluster_centers[3:]).flatten()
    #print positive_centers, negative_centers


    ## cluster all data into 6 groups depending on which centroid its closest to
    ## e.g. 0.21 will be closest to centroid 0.22 as opposed to 0.08
    finalClusters = []

    def getClusters(sentiCenter, senti_tweets):
        cluster1 = []
        cluster2 = []
        cluster3 = []

        for i in range(len(senti_tweets)):

            dist1 = math.fabs(sentiCenter[0] - senti_tweets[i])
            dist2 = math.fabs(sentiCenter[1] - senti_tweets[i])
            dist3 = math.fabs(sentiCenter[2] - senti_tweets[i])

            # + str(emotweetIDs[i])+","
            if (dist1 > dist2):
                if (dist2 > dist3):
                    # cluster3.append(str(emotweets[i])+","+ str(targets[emotweetIDs[i]]))
                    cluster3.append(senti_tweets[i])
                else:
                    cluster2.append(senti_tweets[i])
            else:
                if (dist1 > dist3):
                    cluster3.append(senti_tweets[i])
                else:
                    cluster1.append(senti_tweets[i])

        finalClusters.append(cluster1)
        finalClusters.append(cluster2)
        finalClusters.append(cluster3)

    ### get Clusters for positive tweets
    finalClustersDictIdx = 1
    getClusters(positive_centers, pos_tweets)


    ### get Clusters for negative tweets
    finalClustersDictIdx = 4
    getClusters(negative_centers, neg_tweets)



    ### combine positive and negative clusters and assign index of the predicted value to finalClustersIdx
    ### This will be the predicted target
    finalClustersDict = {1:"", 2:"", 3:"", 4:"", 5:"", 6:""}
    finalClustersIdx = []

    k = 0
    for clusters in finalClusters:
        finalClustersIdx.append([])

        for cluster in clusters:
            myidx = my_cluster_centers.index(min(my_cluster_centers, key=lambda x: abs(x - cluster)))+1
            #print cluster, my_cluster_centers,  myidx
            finalClustersIdx[k].append(myidx)
        k += 1


    ### get max value in each cluster. This will be the actual target
    indices = []
    for cluster in finalClustersIdx:
        indices.append(max(cluster,key=cluster.count))

    #print indices
    accuracy = [0]*len(my_cluster_centers)

    k = 0
    for cluster in finalClustersIdx:
        for row in cluster:
            if(row == indices[k]):
                accuracy[k] += 1
        k += 1

    for row in range(len(accuracy)):
        #print  accuracy[row], len((finalClusters[row]))
        accuracy[row] = accuracy[row]/float(len(finalClusters[row]))*100


    #print "Accuracy for individual emotions", accuracy
    print ("Average accuracy", sum(accuracy)/len(accuracy))


def fuzzyCmeans(request):
        return(JsonResponse( getfuzzyCmeans(),status=200, safe=False ))
