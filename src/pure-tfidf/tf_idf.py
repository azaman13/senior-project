'''
Look at this documentation for more collection methods
http://nltk.googlecode.com/svn/trunk/doc/api/frames.html
'''
from nltk.corpus import PlaintextCorpusReader
from nltk.text import TextCollection
from nltk.tokenize import *
from nltk.text import *
import sys
from nltk.tokenize import WhitespaceTokenizer
from pprint import pprint
from nltk import*
from nltk.corpus import stopwords
import sets
from collections import Counter
import math

PUNCTLIST = [',','.',';','?',':','!','(',')','{','}','/','==','-','"']
STOPWORDS = stopwords.words('english')


'''
    Returns a sorted dictionary
    
'''
def sortDict(dict):
    items = [(v, k) for k, v in dict.items()]
    items.sort()	
    items.reverse()
    dict = [(k, v) for v, k in items]
    for key,value in dict:
        print "%s: %s" % (key, value)
    return dict

'''
    Saves the simle and normal tf_idf dictionaries in a file for later use
    params: two dictionaries, simple and normal
'''

def savesTfidf(simpleDict, normalDict):
    textFile=open("tfidf.txt","r+")
    for key,value in simpleDict:
        s = str(key) + "\t" + str(value)+"\n"
        textFile.write(s)
    textFile.write("XXXXXXXXXXXXX XXXXXXXXXXXXXXXXXXXXXXX XXXXXXXXXXXXX"+"\n")
    for key,value in normalDict:
        s = str(key) + "\t" + str(value)+"\n"
        textFile.write(s)

'''
    params: name of the text file which needs to be parsed, and returned a list of parapraphs
    return: a list whose every item is a paragraph (of  type string)
'''
def extractParasInList(name):
    corpuslocation ='/Users/anis/seniorProject/aligned Paragraphs/algebra'
    reader = PlaintextCorpusReader(corpuslocation, '.*\.txt')
    # This gives the list of paragraphs. every paragraph list contains ist of sentences
    # So it is a list of lists. Bunch of sentenses as a list joins together to make  		
    # lists of pararagraph
    pList = []
    paragraphlist =  reader.paras(name) #'simpleTuring.txt'
    numpara = len(paragraphlist)
    for sentlist in paragraphlist:
        numsent = len(sentlist)
     	paraAsAList = []
     	# this loops through all the sentence lists and make them one list
        for i in range(numsent):
        	paraAsAList = paraAsAList + sentlist[i]	
     	paraAsAString = ""
     	for word in paraAsAList:
     		paraAsAString = paraAsAString + word + str(" ")
        pList.append(paraAsAString)
    return pList


'''
Given a text file (having different paragraphs seperated by "\n", this method generates the tfidf ') for 
every word in that paragraph and saves them to a file
'''
def compute_tfidf(text,filename):
    numPara = len(text)
    print "there should be this many para in the text file ", numPara
    
    colList = []
    paragraphWords = []
    for i in range(numPara):
        paragraphWords = word_tokenize(text[i])
        colList.append(paragraphWords)
    collection = TextCollection(colList)
     
    for paraList in colList:
        dict={}
        for term in paraList:
            print term, "has weight: ", collection.tf_idf(term,paraList)
            dict[term]= collection.tf_idf(term,paraList)
        '''
        print "BEFORE  <><><><><<><<>><><><><><><><>><><  ",type(dict)
        for key,value in dict.iteritems():
            print key," ",value
        '''
        d=sortDict(dict)
        print "AFTER SORTED  <><><><><<><<>><><><><><><><>><><  ",type(d)
        textFile=open(filename,"a")
        textFile.write("\n")
        for key,value in d:
            s = str(key) + "\t" + str(value)+"\n"
            textFile.write(s)


'''
Removes == Refereance == types things from the article
'''      
def removeSubtitles(list):
    count =0
    for elem in list:
        if(elem.startswith('=')):
            list.remove(elem)
            count = count +1
    return list

'''
This method removes all lists (2 element list) whose first element 
contains element from the punct list
'''
def filterParagraphList(paragraphList):
	for wordlist in paragraphList:		
		if (wordlist!=[] and wordlist[0] in PUNCTLIST):
			paragraphList.remove(wordlist)
	for wordlist in paragraphList:
		if (wordlist!=[] and wordlist[0] in STOPWORDS):
			paragraphList.remove(wordlist)
	return paragraphList


def extractAllWordsFromPara(simpleFileName,normalFileName):
    simpleParagraphsList = []
    normalParagraphsList = []
    numPara=0
    textFile=open(simpleFileName,"r")
    for line in textFile.readlines():
        itemList = word_tokenize(line)
        simpleParagraphsList.append(itemList)     
    # this is a list of paras with ALL tfidf words selected. format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) list of all words
    S_topWordsFromPara = []
    N_topWordsFromPara = []

    sbreak = []
    for index in range(len(simpleParagraphsList)):
        list = simpleParagraphsList[index]
        if(list ==[]):
            sbreak.append(index)
    sbreak.append(len(simpleParagraphsList))
    for idx in range(len(sbreak)-1):
        S_topWordsFromPara.append(simpleParagraphsList[sbreak[idx]+1:sbreak[idx+1] ])
    # NORMAL Part
    textFile=open(normalFileName,"r")
    for line in textFile.readlines():
        itemList = word_tokenize(line)
        normalParagraphsList.append(itemList)

    nbreak = []
    for index in range(len(normalParagraphsList)):
        list = normalParagraphsList[index]
        if(list ==[]):
            nbreak.append(index)
    nbreak.append(len(normalParagraphsList))
    
    
    for idx in range(len(nbreak)-1):
        N_topWordsFromPara.append(normalParagraphsList[nbreak[idx]+1:nbreak[idx+1] ])
        
    return S_topWordsFromPara,N_topWordsFromPara 

def extractTopWordsFromPara(simpleFileName,normalFileName):
    simpleParagraphsList = []
    numPara=0
    textFile=open(simpleFileName,"r")
    for line in textFile.readlines():
        itemList = word_tokenize(line)
        simpleParagraphsList.append(itemList)
        # This removes all the punctuations and other common stopwords from the list. 
        # SEE PUNCTLIST,STOPWORDS for detail
        simpleParagraphsList = filterParagraphList(simpleParagraphsList)
        # this is a list of paras with ALL tfidf words selected. 
        # format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) list of all words
    TOP = 15
    S_topWordsFromPara = []
    N_topWordsFromPara = []
    for index in range(len(simpleParagraphsList)):
        list = simpleParagraphsList[index]
        if(list ==[]):
            if(index+TOP+1<len(simpleParagraphsList)):
                S_topWordsFromPara.append(simpleParagraphsList[index+1:index+TOP+1])
    normalParagraphsList = []

    textFile=open(normalFileName,"r")
    for line in textFile.readlines():
        itemList = word_tokenize(line)
        #simpleTupleList.append(itemList)
        normalParagraphsList.append(itemList)

    
    #This removes all the punctuations and other common stopwords from the list. SEE PUNCTLIST,STOPWORDS for detail
    
    normalParagraphsList = filterParagraphList(normalParagraphsList)    
    # this is a list of paras with TOP tfidf words selected. format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) list of top words
    N_topWordsFromPara = []
    for index in range(len(normalParagraphsList)):
        list = normalParagraphsList[index]
        if(list ==[]):
            if(index+TOP+1<len(normalParagraphsList)):
                N_topWordsFromPara.append(normalParagraphsList[index+1:index+TOP+1])
    return S_topWordsFromPara,N_topWordsFromPara 




'''
Computes dot product of two given lists of equal length
'''

def computeDotProduct(list1,list2):
    result = 0.0
    for i in range(len(list1)):

        result = result + (float(list1[i]) * float(list2[i]))
    return result



'''
    Given Two Paragraphs, this returns the list if common shared words. Params: para1 = para2 = [ ...[   ,  ] , [   ,   ], [   ,   ] ......      ] 
'''

def findCommonWords(Para1,Para2):
    sharedWords = []
    simpleWeights = []
    normalWeights = []
    
    for tuple1 in Para1:
        for tuple2 in Para2:
            if(tuple1!=[] and tuple2!=[] and tuple1[0]==tuple2[0]):
                sharedWords.append(tuple1[0])
                simpleWeights.append(tuple1[1])
                normalWeights.append(tuple2[1])
    dotProduct = computeDotProduct(simpleWeights,normalWeights)
    return sharedWords,dotProduct

'''
    This uses "findCommonWords" method which in turn uses "computeDotProduct" method
'''

def matchingPara(simpleDoc,normalDoc):
    big = []
    for sPara in simpleDoc:

        commonWord = []
        max = 0
        whichPara=0
        list = []

        for nPara in normalDoc:
            #print "These are the shared words between the simple and normal",simpleDoc.index(sPara),normalDoc.index(nPara)
            
            commonWord, dotProduct = findCommonWords(sPara,nPara)
            if(dotProduct>=max):
                max = dotProduct
                whichPara = normalDoc.index(nPara)
                list = commonWord
        pairedPara[simpleDoc.index(sPara)] = whichPara
    
    

def formSentenceList(spara,npara):
    col = []
    sslist = sent_tokenize(spara)

    nslist = sent_tokenize(npara)
    #print sslist
    #print nslist
    
    for sentence in sslist:
        tokenizedSentence = word_tokenize(sentence)
        col.append(tokenizedSentence)
    
    for sentence in nslist:
        tokenizedSentence = word_tokenize(sentence)
        col.append(tokenizedSentence)
    return col,sslist,nslist

def sentenceAlignment(simpleParas, normalParas, pairedPara): 
    #print simpleParas, len(simpleParas)
    #print normalParas, len(normalParas)
    for key,value in pairedPara.items(): # key is simple and value in normal
        print "**********************************"
        print "PARAGRAPH"
        print "##################################"
        SPara = simpleParas[key]
        NPara = normalParas[value]
        # given two paragraphs, it returns a list of all the sentences where each sentence is a list of words, with a list of simple sentence list and normal sentence list
        colList, sslist,nslist = formSentenceList(SPara,NPara)
        #print colList, type(colList), len(colList)
        collection = TextCollection(colList)
        #print collection
        '''
        dict={}
        for sentence in colList:
            weight = 0
            
            for term in sentence:
                weight = collection.tf_idf(term,sentence)
                #print "TERM -> ",term, "is",weight
                # what if the term is already in the dic, we need to add the weight
                if(term not in dict):
                    dict[term] = weight
                # dict[term] = weight
            
            #dict = sortDict(dict)
            
        #print dict
        # Now I have the tf-idf value of every terms in the sentences
        
        '''
        #print sslist
        #print nslist
        temp=[]
        for sentence in sslist:
            tokSen = word_tokenize(sentence)
            temp.append(tokSen)
        sslist = temp
        temp=[]
        for sentence in nslist:
            tokSen = word_tokenize(sentence)
            temp.append(tokSen)
        nslist = temp
        #print sslist
        #print nslist
        ############
        for simpleLine in sslist:
            maxSimilarity = 0
            for normalLine in nslist:
                wordUnion = union(simpleLine,normalLine)
                #print "<><><><<><><><><><><<><><<><>><><"
                #print "SIMPLE"
                #print simpleLine
                #print "NORMAL"
                #print normalLine
                #print "Union"
                #print wordUnion

                simpleVector = []
                normalVector = []
                
                for term in wordUnion:
                    StermFreq = frequecyOccurance(term,simpleLine)
                    NtermFreq =frequecyOccurance(term,normalLine) 
                    simpleVector.append(StermFreq)
                    normalVector.append(NtermFreq)
                
                # now compute the dot product of the two vectors
                #print "Simple Vetor",simpleVector
                #print "Normal Vector",normalVector
                dotProd = computeDotProduct(simpleVector,normalVector)
                #print "Dot product",dotProd
                absSimpleVector = magnitude(simpleVector)
                absNormalVector = magnitude(normalVector)
                normalizedFactor = absSimpleVector*absNormalVector
                similarity = dotProd/normalizedFactor
                if(similarity>maxSimilarity):
                    maxSimilarity = similarity
                    simpleSentence = simpleLine
                    normalSentence = normalLine
            print "<><><><<><><><><><><<><><<><>><><"
            print "similarity=",maxSimilarity
            print simpleSentence
            print normalSentence
        ## END OF ONE PAIRED PARA
        
# finds the magnitude of a vector i.e a list
def magnitude(List):
    result = 0
    for i in List:
        result = result + (i*i)
    return math.sqrt(result)

# returns the frequency of the term in the given list
def frequecyOccurance(term,List):
    d = dict(Counter(List))
    result = 0
    for key,value in d.items():
        if(key==term):
            result = value
            break
    return result




def union(simpleLine,normalLine):
    unionSet = []
    simpleSet = sets.Set(simpleLine)
    normalSet = sets.Set(normalLine)
    temp = simpleSet.union(normalSet)
    unionSet = list(temp)
    return unionSet




pairedPara = {}
simpleParas = []
normalParas = []
orgSimpleParas = []
orgNormalParas = []

if __name__=='__main__':
    simpleParas = extractParasInList('simpleAlgebra.txt')
    normalParas = extractParasInList('normalAlgebra.txt')
    print len(simpleParas)
    print len(normalParas)
    removeSubtitles(simpleParas)
    removeSubtitles(normalParas)
    
    orgSimpleParas = simpleParas
    orgNormalParas = normalParas

    SP = []
    for para in simpleParas:
        words = word_tokenize(para)
        for term in words:
            if (term in STOPWORDS or term in PUNCTLIST):
                words.remove(term)        
        temp = " "
        for word in words:
            temp = temp + str(word)+str(" ")
        SP.append(temp)

    simpleParas = SP
    

    NP = []
    for para in normalParas:
        words = word_tokenize(para)
        for term in words:
            if (term in STOPWORDS or term in PUNCTLIST):
                words.remove(term)        
        temp = " "
        for word in words:
            temp = temp + str(word)+str(" ")
        NP.append(temp)

    normalParas = NP

    #print normalParas
    #print len(simpleParas)
    #print len(normalParas)
    
    #call it one already, I comment the writing line, remember!!!
    #compute_tfidf(normalParas,"normalAlgebratfidf.txt")
    #compute_tfidf(simpleParas,"simpleAlgebratfidf.txt")

    #allSimpleWords,allNormalWords = extractAllWordsFromPara("simpleAlgebratfidf.txt","normalAlgebratfidf.txt")
    topSimpWords,topNormalWords = extractTopWordsFromPara("simpleAlgebratfidf.txt","normalAlgebratfidf.txt") # based on paragraphs
    #matchingPara(allSimpleWords,allNormalWords)
    matchingPara(topSimpWords,topNormalWords)

    sentenceAlignment(orgSimpleParas,orgNormalParas, pairedPara)
    # Now align sentences including wordnet info at a phrase level
