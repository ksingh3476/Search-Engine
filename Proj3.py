import os
from bs4 import BeautifulSoup
from Tkinter import *
import webbrowser
import nltk, re, pprint
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords

import pymongo
from pymongo import MongoClient

import string
import re
from collections import defaultdict
import json
import math

#README
#Group: 90
#Josh Lu, ID: 60343957
#Bryan Fullam, ID: 30047908
#Ketan Singh, ID: 78773036


#got from https://stackoverflow.com/questions/18262293/how-to-open-every-file-in-a-folder

TOTAL_DOC_COUNT = 37497
FINALLINKS = []
"""
Take the path name, and extract the ID of the documents, get rid of the content from the script and style headers
Out of the content we get only the alphanumeric words. Optimize tokens gets rid of stop words and unecessary words.
"""
def create_index(path, index_database):
    temp_database = defaultdict(list)
    for root, dir, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(file_path)
            doc_id = file_path.split('/')
            doc_id = doc_id[2:4]
            print(doc_id)
            
            if file_path != "./WEBPAGES_RAW/bookkeeping.json" and file_path != "./WEBPAGES_RAW/bookkeeping.tsv":
                
                f = open(file_path,"r")
                soup = BeautifulSoup(f.read(), "lxml")
                #  soup = BeautifulSoup(f).get_text(" ", strip = True)
                
                #soup.encode("ascii", "ignore")
                selects = soup.findAll(['script', 'style'])                 #don't need these tags
                for match in selects:
                    match.decompose()
            
                soup = soup.get_text(" ", strip = True)
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(soup)
                tokens = [w.lower() for w in tokens]
                
                optimized_tokens = optimize(tokens)
                #   print(optimized_tokens)
                
            input_index(optimized_tokens, doc_id, temp_database)        #create index with only id, count, tf

    include_tf_idf(index_database, temp_database)                           #after all docs parsed, update with tf-idf



#________________STOP WORDS_________________________

            #words we can get rid of:
            #           http
            #           https
            #           yf\xfe\xea  ?
"""
This function gets rid of stop words in order to optimize our tokens, and words greater than 50 in
letters of len 1. And alphanumeric characters only.
"""

def optimize(tokens):
    words = [word for word in tokens if word.isalnum()]         #https://machinelearningmastery.com/clean-text-machine-learning-python/


    regex = r"([^A-Za-z0-9]+)"                  #filter out all non alphanumeric

    
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "http", "https"]

    updated_tokens = [word for word in words if not re.search(regex, word) and len(word) != 1 and  len(word) < 50 and not word in stop_words]

    

        #and word != 'http' and word != 'https'
    
    return updated_tokens



#_________________________________________

"""
Go into token list and put every word into a temporary dictionary, and then include the count of each term.
Use this temporary dictionary to create the actual index, make the count and find the normalized term 
frequency
"""


def input_index(tokens, doc_id, database):
    global TOTAL_DOC_COUNT
    if len(doc_id) == 2:
        id = str(doc_id[0] + '/' + doc_id[1])
        temp_dict = {}
        for token in tokens:                                    #counting the frequency, for each term
            if token not in temp_dict:
                temp_dict[token] = 1

            else:
                temp_dict[token] += 1

        for key, value in temp_dict.iteritems():
            print("_______")
            tf = float(value / float(len(tokens)))
            print(tf)
            term_freq = math.log10(tf)
            print(term_freq)
            term_frequency = float(1 + float(math.log10(value) / len(tokens)))                   #normalized tf, tokens is total terms in this doc

            string_value = str(id) + "," + str(value) + "," + str(term_frequency)               #convert info into a string
            
                                                                                              
            database[key].append(string_value)                                                  #append to temp database





#__________________________________________
"""
Go back into that data structure, calculate the TF idf and then we update the index
"""

def include_tf_idf(new_database, old_database):

    for key, value in old_database.iteritems():
        new_value = []
        for posting in value:                                                   #multiple postings will have to update each
            temp = posting.split(",")                                           #convert data back into normal form
            tf = float(temp[2])                                                 #we get the tf number
            #print(len(value))
            tf_idf = tf * (math.log(TOTAL_DOC_COUNT / len(value)))              #determin tf_idf
            posting = posting + "," + str(tf_idf)                               #update the new data for each doc
                
            new_database[key].append(posting)                                   #put it in the new final database




#__________________________________________
#writing the index database to a file
"""
This function writes the index we created into a text file for future use
"""

def write_results(index_database):
    print("WE ARE STARTING THE WRITE")
    f = open ('result.txt', 'w')                                #write into result.txt
    for k, v in index_database.items():
        str_value = ""                                          #append all the postings of a term together into a big string
        count = 0
        for x in v:
            #print(x)
            if count == 0:
                str_value += x
                count += 1
            else:
                if count < len(v):
                    str_value += "_"
                    str_value += x
                    count += 1
                else:
                    str_value += x

        f.write( k + ' >>> '+ str_value + '\n')             #write into the file

 
    f.close()





#________________________________________
# we will rank the documents based on the query

"""
Extract index, takes the written text from before and writes this into a dictionary
"""

def extract_index():

    with open('result_final.txt', 'r') as document:
        index = defaultdict(list)
        for line in document:
            line = line.split(">>>")                                #we convert all the postins back into desired form
            term = line[0].strip()
            line = line[1:]
            for x in line:
                y = x[1:]
                y = y.split("_")
            if not line:  # empty line
                continue
            index[term] = y

    return index                                                 #return the dict that we will use for search engine








#________________________________________
# we will rank the documents based on the query

"""
Acts as the main runner for the search engine given the query
has some GUI Elements as well
"""

def search_engine(index,sq):
    sq = str(sq)
    if sq == "":
        root3 = Tk()
        mylabel2 = Label(root3, text = "NO RESULTS FOUND").pack()
        root3.mainloop()
    else:
        search_query = sq.split(" ")                                          #if multiple queries, split into multiple parts
        [x.lower() for x in search_query]
    
        topten = search_index(index,search_query)


    
        top_ten = search_index(index, search_query)                                     #now find top 10 results and return
        format_results(top_ten, search_query)




#________________________________________

"""
helper function to bind buttons with urls
"""
def callback(url):
    webbrowser.open("http://"+url)

"""
format results takes the top ten links provided and prints them to the console and GUI
"""

def format_results(answers, search):
    with open('./WEBPAGES_RAW/bookkeeping.json') as f:
        data = json.load(f)
       
    our_search = " "
    for x in search:
        our_search += x
        our_search += " "

    print("Top Ten Results for:" + our_search)
    count2 = 1
    root2 = Tk()
    if answers != None and len(answers) > 0:
        mylabel = Label(root2,text = "Search results for: " + our_search).pack()
        buttons = []
        for x in range (len(answers)):
            y = int(x);
            mybtn1 = Button(root2, text = data[answers[y]], command = lambda y=y : callback(data[answers[y]])).pack()
            buttons.append(mybtn1)
        for x in answers:
            print("{}. ".format(count2) + x + " = " + data[x])
            count2+=1
    else:
        mylabel2 = Label(root2, text = "NO RESULTS FOUND").pack()

    root2.mainloop()





#________________________________________
# this is to do the search for the three keywords required in the assignment
# we will print the top 10 results for each search
"""
this function finds the term postings of the query and ranks them and returns the list of highest ranked lists
one condition for single word queries and another for multiple word queries
"""

def search_index(index, term_list):

    global TOTAL_DOC_COUNT
    
    if len(term_list) == 1:                                      # if only one word in the search query

        for term in term_list:
            value = index[term.lower()]                                 # find the postings affiliated with this term
            
            if len(value) > 0:
                
                results = []
                for x in value:
                    x = x.split(",")                            #convert data back into normal form
                    results.append(x)
                
                
                
                sorted_by_tfidf = sorted(results, key=lambda x: float(x[3])  , reverse=True)         # sorting by tf*idf, last data value in each posting
                sorted_by_tfidf = sorted_by_tfidf[0:10]                                             #get top 10 results
                
                final = []
                for x in sorted_by_tfidf:                                                           #we return only the doc ids
                    final.append(x[0])
                

                return final
            
            else:
                print("NO RESULTS")


    elif len(term_list) > 1:                             #if there are multiple words in search queries
        query_ids = []                                  #gonna have to do mulitple searched, and combine results
        query_results = []
        term_dict = {}
        
        for term in term_list:
            value = index[term.lower()]
            
            if len(value) > 0:
                
                results = []
                for x in value:
                    x = x.split(",")
                    results.append(x[0])
                    if x[0] not in term_dict.keys():
                        term_dict[x[0]] = float(x[3])
                    else:
                        term_dict[x[0]] += float(x[3])
                
                query_ids.append(results)                                     #group all query results together
            
            else:
                print("NO RESULTS")


        if len(query_ids) > 0:
            for number in range(0, len(query_ids)):
                if number == 0:
                    s1 = set(query_ids[number])
                    temp_intersection = s1
                
                elif number > 0:
                    s1 = set(query_ids[number])
                    #print(s1)
                    temp_intersection = temp_intersection.intersection(s1)


        
            combine_ranks = []
            for x in temp_intersection:
                temp = []
                temp.append(x)
                score = 0.0
                temp.append(term_dict[x])
                combine_ranks.append(temp)


            sort_answers = sorted(combine_ranks, key=lambda x: float(x[1]) , reverse=True)

            sortlist = [x[0] for x in sort_answers[0:10]]                       #return the top ten results
    





            outlist = []
            outlist.extend(sortlist)
            FINALLINKS = list(temp_intersection)[0:10]
            return outlist                               #return the top ten results



#--------------------------- GUI-----------------------------------------------





if __name__ == "__main__":
    
    '''
        
    index = defaultdict(list)                                 #uncommented out stuff is generating the index
    path= './WEBPAGES_RAW'
    tokens = create_index(path, index)
    
    
    #print(index)
    
    write_results(dict(sorted(index.items())))
    print(len(index))
    '''

    # print(extracted_index)
    print("WELCOME TO CS121 SEARCH ENGINE")                         #this is the start of the actual search engine
    print("loading index ... ")
   
    extracted_index = extract_index()
    root = Tk()
    frame = Frame(root)
    frame.pack()
    mylabel = Label(root,text = "CS 121 Search Engine").pack()
    searchq = StringVar()
    myentry = Entry(root,textvariable = searchq).pack()
    #toplinks = search_engine(extracted_index,searchq.get())
    mybutton = Button(root, text = "search",command = lambda: search_engine(extracted_index,searchq.get())).pack()
    
   
   
    
    root.mainloop()





