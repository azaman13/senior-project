'''
    Look at this documentation for more collection methods
http://nltk.googlecode.com/svn/trunk/doc/api/frames.html
'''
from nltk.corpus import PlaintextCorpusReader
from nltk.text import TextCollection
from nltk.tokenize import *#sent_tokenize import word_tokenize
from nltk.text import *
import sys
from nltk.tokenize import WhitespaceTokenizer
from pprint import pprint
from nltk import*
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


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
    # So it is a list of lists. Bunch of sentenses as a list joins together to make  		#lists of pararagraph
    pList = []
    paragraphlist =  reader.paras(name) #'simpleTuring.txt'
    numpara = len(paragraphlist)
    for sentlist in paragraphlist:
        #print sentlist
        numsent = len(sentlist)
        #print type(sentlist),
        #print numsent
     	paraAsAList = []
     	# this loops through all the sentence lists and make them one list'''
        for i in range(numsent):
        		paraAsAList = paraAsAList + sentlist[i]	
        #print paraAsAList # this is the whole parapragph as one list
     	paraAsAString = ""
     	for word in paraAsAList:
        		paraAsAString = paraAsAString + word + str(" ")
        #print paraAsAString
        pList.append(paraAsAString)
        #print len(pList)
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
    
    #for i in colList:
        #print i
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
        '''
        for key,value in d:
            print key," ",value
        '''
        textFile=open(filename,"a")
        textFile.write("\n")

        for key,value in d:
            s = str(key) + "\t" + str(value)+"\n"
            #print s
            textFile.write(s)
        #print dict
    

'''
Removes == Refereance == types things from the article
'''      
def removeSubtitles(list):
    count =0
    for elem in list:
        #print "before ", list
        #print elem, list.index(elem)
        if(elem.startswith('=')):
            #print elem
            list.remove(elem)
            count = count +1
        #print "after ", list
    
    #print str("final list "),
    print "removed that many lines ", count
    return list

'''
This method removes all lists (2 element list) whose first element contains element from the punct list
'''
def filterParagraphList(paragraphList):
	
	for wordlist in paragraphList:		
		#print wordlist
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
    #print simpleParagraphsList
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
        # This removes all the punctuations and other common stopwords from the list. SEE PUNCTLIST,STOPWORDS for detail
        simpleParagraphsList = filterParagraphList(simpleParagraphsList)
        
        # this is a list of paras with ALL tfidf words selected. format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) list of all words
    #print simpleParagraphsList
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
        #print itemList
        normalParagraphsList.append(itemList)

    
    #This removes all the punctuations and other common stopwords from the list. SEE PUNCTLIST,STOPWORDS for detail
    
    normalParagraphsList = filterParagraphList(normalParagraphsList)    
    # this is a list of paras with TOP tfidf words selected. format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) list of top words
    N_topWordsFromPara = []
    for index in range(len(normalParagraphsList)):
        list = normalParagraphsList[index]
        if(list ==[]):
            #print index, "   ", list
            if(index+TOP+1<len(normalParagraphsList)):
                #print normalParagraphsList[index+1:index+TOP+1]
                N_topWordsFromPara.append(normalParagraphsList[index+1:index+TOP+1])
                #print "\n"
                
    #print len(N_topWordsFromPara)
                
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
                #print tuple1[1],tuple2[1]
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
        print "\n"
        print "The", whichPara, "th para from normal matches with the", simpleDoc.index(sPara),"th para from simple, with dot product = ", max, "with shared words ", list
        print "SIMPLE "
        print orgSimpleParas[simpleDoc.index(sPara)], "\n"
        print "NORMAL "
        print orgNormalParas[whichPara]
        pairedPara[simpleDoc.index(sPara)] = whichPara
    
    


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
    print pairedPara
