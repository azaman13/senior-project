'''
Author: Anis Zaman
Bard College, Senior Project
Date: Spring 2013

Description: 

'''
from nltk import *
from nltk.corpus import PlaintextCorpusReader
from nltk.text import TextCollection
from nltk.tokenize import *#sent_tokenize import word_tokenize
from nltk.text import *
from nltk.tokenize import WhitespaceTokenizer
from pprint import pprint
from word import *
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet as wn 
from word import *
#import matplotlib.pyplot as plt
import sys
import sets
import math
import itertools
import os
pairedPara = {}
simpleParas = []
normalParas = []
orgSimpleParas = []
orgNormalParas = []
# these will store all the nouns and verbs from the respective sentences
wordsInSimpleSentence = []
WordsinNormalSentence = []
v1 = {}
v2 = {}
n1 = {}
n2 = {}
PUNCTLIST = [',','.',';','?',':','!','(',')','{','}','/','==','-','"']
STOPWORDS = stopwords.words('english')
TAGLIST = ['CC','CD' ,'DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ', 'WDT','WP','WRB']
commonAuxilaryVerbs  = ['is', 'are', 'was', 'were', 'am', 'be', 'been', 'will', 'shall', 'have', 'has', 'had', 'would', 'could', 'should', 'do', 'does', 'did', 'can', 'may','might', 'must', 'seem']

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
    corpuslocation ='/home/aniszaman/seniorProject/combined/carnivore'
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
            #print term, "has weight: ", collection.tf_idf(term,paraList)
            dict[term]= collection.tf_idf(term,paraList)
        '''
        print "BEFORE  <><><><><<><<>><><><><><><><>><><  ",type(dict)
        for key,value in dict.iteritems():
            print key," ",value
        '''
        d=sortDict(dict)
        #compute_tfidfprint "AFTER SORTED  <><><><><<><<>><><><><><><><>><><  ",type(d)
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
        '''
        print "\n"
        print "The", whichPara, "th para from normal matches with the", simpleDoc.index(sPara),"th para from simple, with dot product = ", max, "with shared words ", list
        print "SIMPLE "
        print orgSimpleParas[simpleDoc.index(sPara)], "\n"
        print "NORMAL "
        print orgNormalParas[whichPara]
        '''
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
        #print SPara
        #print "\n"
        #print NPara 
        #print "\n<><><><><<><>"
        
        # given two paragraphs, it returns a list of all the sentences where each sentence is a list of words, with a list of simple sentence list and normal sentence list
        colList, sslist,nslist = formSentenceList(SPara,NPara)
        #print colList, type(colList), len(colList)
        collection = TextCollection(colList)
        #print collection
        ########### this was commented
        
        dict={}
        for sentence in colList:
            weight = 0
            
            for term in sentence:
                weight = collection.tf_idf(term,sentence)
                print "TERM -> ",term, "is",weight
                # what if the term is already in the dic, we need to add the weight
                if(term not in dict):
                    dict[term] = weight
                # dict[term] = weight
            
            #dict = sortDict(dict)
        print "================================================================"    
        #print dict
        # Now I have the tf-idf value of every terms in the sentences
        
        '''
        #print sslist
        #print nslist
        temp=[]
        print sslist
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
                    #print term
                    StermFreq = frequecyOccurance(term,simpleLine)
                    NtermFreq =frequecyOccurance(term,normalLine)
                    #print StermFreq, NtermFreq 
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
            #print simpleSentence
            #print normalSentence
        ## END OF ONE PAIRED PARA
        '''
def parseFile(fileName):
    filePath  = "/home/aniszaman/seniorProject/combined/carnivore/"+fileName
    name = "parsed"+fileName
    outputFile=open(name,"w+")
    outputFile.close()
    cmd = '/home/aniszaman/seniorProject/stanford-parser/lexparser.sh '+filePath+' >'+name

    #print cmd
    #print filePath 
    os.system(cmd)
    
    #print result
    
    #textFile=open(parsedFileName,"a+")
    #textFile.write(result)
'''
If a sentence is not parsable by Stanford parser, i.e after attempting to parse, the parser wrote the following in the 
parsed file: 
Sentence skipped: no PCFG fallback.\n
SENTENCE_SKIPPED_OR_UNPARSABLE\n' 
'''
def verifyParsedFile(filename):
    textFile=open(filename,"r")
    numLines = textFile.readlines()
    if len(numLines)==2:
        if numLines[1] == 'SENTENCE_SKIPPED_OR_UNPARSABLE\n':
            return False 
        return True
'''
Given two paragraphs, this method returns two list containing the simple sentence and normal sentences
'''    
def formSentenceListForSementic(spara,npara):
    col = []
    simpleSentList = []
    normalSentList = []
    sslist = sent_tokenize(spara)

    nslist = sent_tokenize(npara)
    #print sslist
    #print nslist
    
    for sentence in sslist:
        #tokenizedSentence = word_tokenize(sentence)
        #col.append(tokenizedSentence)
        #col.append(sentence)
        simpleSentList.append(sentence)
    for sentence in nslist:
        #tokenizedSentence = word_tokenize(sentence)
        #col.append(tokenizedSentence)
        #col.append(sentence)
        normalSentList.append(sentence)
    return simpleSentList,normalSentList    
def computeSemanticWeight(term, words):
    for word in words:
        if(word.getValue()==term):
            return float(word.getWeight())
        return 0

def computeTfidfWeight(term,Line,wordsWithWeight):
    if(term not in Line):
        return 0
    for w in wordsWithWeight:
        if(w.getValue()==term):
            return w.getWeight()

def listToString (list):
    result = ""
    for w in list:
        result = result+w+" "
    return result
def removeDuplicatedWords(words):
    seen_words = set()
    result = []
    for w in words:
        if w.getValue() not in seen_words:
            result.append(w)
            seen_words.add(w.getValue())
    return result

def lookUp(word, lst):
    for item in lst:
        if(item.getValue()==word):
            return item.getWeight()
'''
pairedPara = dic of which para match with which one
simpleParas, normalParas are dic of paragraphs (key, value) => (para number, paragraph)
'''
def alignText(simpleParas, normalParas, pairedPara): 
    #print simpleParas, len(simpleParas)
    #print normalParas, len(normalParas)
    for key,value in pairedPara.items(): # key is simple and value in normal        
        SPara = simpleParas[key]
        NPara = normalParas[value]
        #print SPara
        #print "----------"
        #print NPara
        print "=================Paragraphs were above======================================"
        print "=================Paragraphs were above======================================"
        print "=================Paragraphs were above======================================"
        print "=================Paragraphs were above======================================"
        print "=================Paragraphs were above======================================"
        # given two paragraphs, it returns a list of all the sentences where each sentence is a list of words, with a list of simple sentence list and normal sentence list
        colList, sslist,nslist = formSentenceList(SPara,NPara)
        #print colList, type(colList), len(colList)
        collection = TextCollection(colList)
        #print collection

        # this is a list of Word object
        wordsWithWeight = []

        dict={}
        for sentence in colList:
            weight = 0
            #print sentence    
            for term in sentence:
                if term not in PUNCTLIST or term not in STOPWORDS or term not in commonAuxilaryVerbs:
                    weight = collection.tf_idf(term,sentence)
                    #print "TERM -> ",term, "is",weight
                    # what if the term is already in the dic, we need to add the weight
                    if(term not in dict):
                        w = Word(term,"","")
                        w.setWeight(weight)
                        wordsWithWeight.append(w)
                        #dict[term] = weight
                    # dict[term] = weight
            
            #dict = sortDict(dict)
            
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

        
        for simpleLine in sslist:
            stringSimpleLine = listToString(simpleLine)
            # semantic part
            simplefilename = "sentence1.txt"
            SFile=open(simplefilename,"w+")
            SFile.write(stringSimpleLine)

            SFile.close()
            parseFile("sentence1.txt")
            # if failed to parse, skip this sentence and continue
            if verifyParsedFile("parsedsentence1.txt")  == False:
                continue

            buildClause("parsedsentence1.txt", "one")
            # end semantic part


            maxSimilarity = 0
            for normalLine in nslist:
                stringNormalLine = listToString(normalLine)
                # semantic part
                normalfilename = "sentence2.txt"
                NFile=open(normalfilename,"w+")
                NFile.write(stringNormalLine)
                NFile.close()
                parseFile("sentence2.txt")
                #check whether parsing was done properly
                # if failed to parse, skip this sentence and continue
                if verifyParsedFile("parsedsentence2.txt")  == False:
                    #print "**********************************"
                    #print "##################################"
                    #print "failed to parse:---> "+ sentence2
                    continue

                # end semantic part

                #buildClause("parsedsentence1.txt", "one")
                #print "=======SENTENCE Two======="
                buildClause("parsedsentence2.txt","two")
                sentence1Words = []
                sentence2Words = []
                #makeContextFile(n1,v1,n2,v2)
                
                sentence1Words, sentence2Words = makeContextFile(n1,v1,n2,v2)
                #print len(sentence1Words), len(sentence2Words)
                
                # all words is a dictionary of words:tfidf. I converted this to a dictionary from a list of wordsWithWeight for convenience 
                allWords = {}
                for w in wordsWithWeight:
                    #w.printWord()
                    allWords[w.getValue()]=w.getWeight()
                #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                #print allWords
                # for first part, sentence 1 words
                numerator1 = 0
                denominator1 = 0
                #print allWords

                for word in sentence1Words:
                    if(word.getValue() in allWords):
                        tfidf = allWords[word.getValue()]
                        semanticWeight = word.getWeight()
                        numerator1 = numerator1+ (semanticWeight*tfidf)
                        denominator1 = denominator1 + allWords[word.getValue()]
                if(denominator1==0):
                    denominator1 = 1
                partA = numerator1/denominator1
                

                #for w in sentence2Words:
                #    print w.getValue(),
                # for second part, sentence 2 words
                numerator2 = 0
                denominator2 = 0
                for word in sentence2Words:
                    #print "dic index:->", word.getValue(),"value: ",allWords[word.getValue()]
                    if(word.getValue() in allWords): 
                        tfidf = allWords[word.getValue()]
                        semanticWeight = word.getWeight()
                        numerator2 = numerator2+ (semanticWeight*tfidf)
                        denominator2 = denominator2 + allWords[word.getValue()]
                if(denominator2==0):
                    denominator2 = 1
                partB = numerator2/denominator2
                

                SIMILARITY = (partA + partB)/2
                print "><><><><><><><><><><><><><><><><><><><><><><"
                print stringSimpleLine
                print "--------------------------------------------"
                print stringNormalLine
                print "Similarity Score -----> ", SIMILARITY
                print "><><><><><><><><><><><><><><><><><><><><><><"

                #print "The similarity is: ",
                #print partA, partB, SIMILARITY
                
        ## END OF ONE PAIRED PARA

def computeWeight(term,Line,wordsWithWeight):
    if(term not in Line):
        return 0
    for w in wordsWithWeight:
        if(w.getValue()==term):
            return w.getWeight()


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

def buildClause(filename,whichSentence):
    
    #result = os.system('/Users/anis/Documents/stanford-parser-2012-11-12/lexparser.sh /data/sample.txt')
    #print result
    textFile=open(filename,"r")
    LIST = []
    sen = ""
    for line in textFile.readlines():
        if(line!="\n"):
            sen=sen+line
            LIST.append(word_tokenize(line))

    # getting rid of the (ROOT(S and the two closing ")" form the end to keep sen balanced
    #print (LIST)
    # this will show the parse tree, uncomment it if you want to see the tree
    #print sen
    if len(LIST)>0:
        LIST = LIST[2:]
    #print (LIST)
    # now removing the two ")" from the last list
        index = len(LIST)-1
        #print "%%%%%%%%%%%"
        #print index, len(LIST), LIST
        lastList = LIST[index]
        lastList = lastList[0:len(lastList)-2]
        LIST = LIST[0:len(LIST)-1] 
        LIST.append(lastList)
    # this conver list of lists (i.e LIST) to a one flat list, LIST
        LIST = list(itertools.chain.from_iterable(LIST))
    
    #print LIST
    
    count = 0
    clause = ""
    MAP = []

    for index in range(len(LIST)):
        item = LIST[index]
        if(item=="("):
            count = count + 1
            
        if(item==")"):
            count= count - 1
        
        if(count==0):
            MAP.append(index)

    #print MAP 

    #print len(LIST)
    

    NOUN=[]
    VERB=[]
                
    for i in range(len(MAP)):
        
        if(i==0):
            #print "NOUN"
            NOUN = LIST[i:MAP[i]+1]
            #print NOUN
        
        if(i==len(MAP)-1):
            #print "LAST PART"
            #print LIST[MAP[i]: ]
            #print i, MAP[i], MAP[i-1]
            #print LIST[(MAP[i-1]+1) : ]
            print " "
        # the middle part
        if(i>0 and i < len(MAP)-1):
            #print i,MAP[i]
            start = MAP[i-1]+1
            end = MAP[i]
            #print "VERB      "
            VERB =LIST[  start   :   end   ]
            #print VERB
            
    #print "NOUN PHRASE><><><><><><><><"
    nounPhrase = ""
    importantNouns={}
    for index in range(len(NOUN)):
        if(index+1<=(len(NOUN)-1) and index-2 > 0):
            
            if(NOUN[index]== ")" and NOUN[index-2] in TAGLIST ):
                #print NOUN[index-1],NOUN[index-2],
                nounPhrase = nounPhrase + NOUN[index-1]+ " "+ NOUN[index-2] + " "
                if(NOUN[index-2].startswith('N')):
                    importantNouns[NOUN[index-1]] = NOUN[index-2]


    #print "NOUN PHRASE as a string><><><><><><><><><><"
    #print nounPhrase
    #print "IMPORTANT NOUNS"
    #print importantNouns
    if(whichSentence=="one"):
        global n1
        n1 = importantNouns
    elif(whichSentence=="two"):
        global n2
        n2 = importantNouns

    #print NOUN
    #print "\n"
    #print "VERB PHRASE: <><><><>"
    verbPhrase = ""
    importantVerbs = {}
    for index in range(len(VERB)):
        if(index+1<=(len(VERB)-1) and index-2 > 0):
            
            if(VERB[index]== ")" and VERB[index-2] in TAGLIST ):
                #print VERB[index-1],VERB[index-2],
                verbPhrase = verbPhrase + VERB[index-1]+ " "+ VERB[index-2] + " "
                # filtering out the auxilary verbs and taking only the interesting verbs
                if((VERB[index-1] not in commonAuxilaryVerbs) and VERB[index-2].startswith('V')):
                    importantVerbs[VERB[index-1]] = VERB[index-2]
    #print "VERB PHRASE as a string <><><><><><><><>"
    #print verbPhrase
    #print "IMPORTANT VERBS"
    #print importantVerbs

    if(whichSentence=="one"):
        global v1
        v1 = importantVerbs
    elif(whichSentence=="two"):
        global v2
        v2 = importantVerbs

'''
    This method will run the ukb personalized page ranking to get the sense of every word
    @params:
        filename -> name of file where the important nouns and verbs of the simple and normal sentence exist
        
'''
def extractSense(fileName):
    #print "****calling ukb-2****"
    binFile = "/home/aniszaman/seniorProject/ukb-2.0/bin/wn30.bin"
    dicFile = "/home/aniszaman/seniorProject/ukb-2.0/lkb_sources/30/wnet30_dict.txt"
    # finally this will change
    inputFile = "/home/aniszaman/seniorProject/combined/carnivore/"+fileName
    ukbPath  = "/home/aniszaman/seniorProject/ukb-2.0/bin/ukb_wsd"
    wordSenseFileName = "sensed"+fileName
    # just creating the file, otherwise, the result gets concatenated 
    f=open(wordSenseFileName,"w+")
    f.close()

    command = ukbPath+  " --ppr -K "+binFile+" -D "+dicFile+" "+inputFile+ '>' + wordSenseFileName
    os.system(command)

def filterWords(allWords):
    for word in allWords:
        if(word.getWeight()<0):
            allWords.remove(word)
    return allWords

'''
    read the sense file and update the wsd of every word of the sentence1 & sentence2 -> list of Word objects.
'''
def readWSD(wsdFile, sentence1, sentence2):

    textFile=open(wsdFile,"r")
    lines = []
    for line in textFile.readlines():
        lines.append(line)
        
    del lines[0]
    #lines look like this: ['ctx_01 w2  02684924-v !! continue\n',.... 'ctx_02 w1  01056411-n !! stop\n', ]
    #for l in lines:
    #    print l
    #print "#############################WORDS FROM SENTENCE1###################################"
    #for w in sentence1:
    #    print w.getValue(),w.getRootValue()
    #print "##############################################################"
    for l in lines:
        r = l.split()
        if(r[0]=='ctx_01'):
            # will work on sentence1
            senseVal  = []
            root = r[len(r)-1]
            senseVal = r[len(r)-3]
            #print root, senseVal
            #print type(senseVal)
    
            #print root, senseVal
            #print type(senseVal)
            # this loop gets rid of any precidding 0 from the database location/ synset location
            while(senseVal[0]=="0"):
                senseVal = senseVal[1:]
            #print root, senseVal
            #print type(senseVal)
        
            for w in sentence1:
                #print w.getValue()
                if(w.getRootValue()==root and isinstance("",type(senseVal))):
                    #print w.getRootValue(), root

                    # splitting '13244109-n'  ---> ['13244109', 'n']
                    senseVal = senseVal.split('-')
                    p = senseVal[len(senseVal)-1]
                    num = int(senseVal[0])
                    #print p, num, w.value
               
                    if(num!=-1):
                        s = wn._synset_from_pos_and_offset(p, num)
                        
                        s = s.__str__()
                        s = s.split("'")[1] 
                        w.setSynSet(s)
        
        if(r[0]=='ctx_02'):
            root2 = r[len(r)-1]
            senseVal2 = r[len(r)-3]
            #print "1=>", type(senseVal2)
            #s2SenseVal = r[len(r)-3]
            #print "------>",root, senseVal
            # this loop gets rid of any precidding 0 from the database location/ synset location
            while(senseVal2[0]=="0"):
                senseVal2 = senseVal2[1:]
            #print "------>>>>>>>>",root, senseVal
            #print "2=>", type(senseVal2)
            #print senseVal2
            for w in sentence2:
                if(w.getRootValue()==root2 and isinstance("",type(senseVal2))):
                    #print w.getRootValue(), root2, senseVal2, type(senseVal2)
                    #print "before", len(senseVal2), type(senseVal2),senseVal2
                    senseVal2 = senseVal2.split('-')
                    #print "AFTER:", senseVal2

                    p = senseVal2[len(senseVal2)-1]
                    num = int(senseVal2[0])

                    #print w.getRootValue(), p, num, w.value
                    if(num!=-1):
                        s = wn._synset_from_pos_and_offset(p, num)
                        s = s.__str__()
                        s = s.split("'")[1]
                        w.setSynSet(s)
                        
        
    #print "^^^^^^^^^^^^^^^^^^^^^^^^"
    SENT1NOUNS = []
    SENT1VERBS = []
    
    SENT2NOUNS = []
    SENT2VERBS = []

    for w1 in sentence1:
        if(w1.getPos()=='NOUN'):
            SENT1NOUNS.append(w1)
        if(w1.getPos()=='VERB'):
            SENT1VERBS.append(w1)
    #    w1.printWord()
    #print "^^^^^^^^^^^^^^^^^^^^^^^^^"
    
    for w2 in sentence2:
        if(w2.getPos()=='NOUN'):
            SENT2NOUNS.append(w2)
        if(w2.getPos()=='VERB'):
            SENT2VERBS.append(w2)
    #    w2.printWord()
    #print "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}"

    # Finding the weights for nouns
    # record the max similarity values, later just add them
    maxNounSimilarityValues = []
    NOUNLIST = []
    VERBLIST = []
    # this is for me to see which word from sentence 1 maps to the corresponding words in sentence 2
    nounMapping = {}
    
    for n1 in SENT1NOUNS:
        #print n1.getValue()
        if(n1.getSynSet()!="null"):
            #n1.printWord()
            exp1 = n1.getSynSet() # this is something like this: believe.v.01
            #print exp1 
            noun1 = wn.synset(exp1)

            maxValue = 0
            # just for printing
            ntemp = ""
            for n2 in SENT2NOUNS:
                if(n1.getSynSet()!="null"):
                    exp2 = n2.getSynSet() # this is something like this: believe.v.01
                    #print exp2
                    #print "this is the expression for noun2==============================================================>>>>>>>"
    
                    #print exp2, type(exp2)
                    if(exp2!='null'):
                        noun2 = wn.synset(exp2)
                        value = noun1.path_similarity(noun2)
                        #print value
                        if(value > maxValue):
                            maxValue = value

            n1.setWeight(maxValue)
            
        else: # if the synset is null
            n1.setWeight(1)    
    # record the max similarity values, later just add them
    
    maxVerbSimilarityValues = []
    # this is for me to see which word from sentence 1 maps to the corresponding words in sentence 2
    verbMapping = {}
    
    #for w in SENT1VERBS:
    #    w.printWord()

    for v1 in SENT1VERBS:
        if(v1.getSynSet()!="null"):
            #v1.printWord()
            exp1 = v1.getSynSet()
            verb1 = wn.synset(exp1)

            maxValue = 0 
            # just for printing
            vtemp = ""
            for v2 in SENT2VERBS:
                if(v2.getSynSet()!="null"):
                    exp2 = v2.getSynSet()
                    verb2 = wn.synset(exp2)
                    value = verb1.path_similarity(verb2)
                    #print value
                    if(value > maxValue):
                        maxValue = value
                        #maxVerbSimilarityValues.append(maxValue)
                        #verbMapping[v1.getValue()] = v2.getValue()
                        #vtemp = v2.getValue()
                        #matchedVerb2 = v2.copy()
            v1.setWeight(maxValue)
            #matchedVerb2.setWeight(maxValue)
            #wordsFromSentence1.append(v1)
            #VERBLIST.append(v1)
            #VERBLIST.append(matchedVerb2)
            #print "the max similarity between " + v1.getValue() + " and "+vtemp +"is" + str(maxValue)
        else:
            v1.setWeight(1)
    allWordsfromSentence1 = SENT1NOUNS+SENT1VERBS
    #print "------------------>>>>>>>>>>>>>>>>>>>>>>>>after>>>>>>>>> sentence1 verbs"
    '''
    for w in SENT1VERBS:
        w.printWord()
    print "------------------>>>>>>>>>>>>>>>>>>>>>>>>after>>>>>>>>> sentence1 nouns"

    for w in SENT1NOUNS:
        w.printWord()
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$TOGETHER$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    for w in allWordsfromSentence1:
        w.printWord()
    '''
    return allWordsfromSentence1
    '''
    #return VERBLIST+NOUNLIST
    return wordsFromSentence1
    #PAIREDSENTENCES.write(res)
    '''
'''
makeContextFile -> this takes in the dic of Noun and Verbs (word:pos format) of two sentences and generate the 
                    input file that is used to get the sense of every word.
    it calls the extractSense method, runs the ukb ppr and save the content in a file  '''
def makeContextFile(nounDic1, verbDic1,nounDic2,verbDic2):
    filename = "context.txt"
    file=open(filename,"w+")
    sentence1 = nounResult1 = verbResult1 =  ""
    sentence2 = nounResult2 = verbResult2 =  ""
    nounWordId1 = verbWordId1 = nounWordId2 = verbWordId2 =  1
    tag = "#"

    # this is a list of Word objects.
    wordsInSentence1 = []
    # noun phrase of sentence 1
    file.write("ctx_01")
    file.write("\n")
    for key,value in nounDic1.items():
        #print key, value
        wordValue = key
        abbPos = value
        if(value[0]=='N'):
            wordPos = "NOUN"
            rootValue = wn.morphy(key,wn.NOUN)
            if(rootValue is None):
                rootValue = key
        WORD = Word(wordValue,wordPos,abbPos)
        WORD.setRootValue(rootValue)
        wordsInSentence1.append(WORD)
        #WORD.printWord()
        nounResult1 = nounResult1+str(WORD.getRootValue())+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId1)+str(tag)+str(1)+" "
        #nounResult1 = nounResult1+str(key)+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId1)+str(tag)+str(1)+" "
        nounWordId1+=1
    #print nounResult1
    
    # verb phrase of sentence 1
    for key,value in verbDic1.items():
        #print key, value
        wordValue = key
        abbPos = value
        if(value[0]=="V"):
            wordPos = "VERB"
            rootValue = wn.morphy(key,wn.VERB)
        WORD = Word(wordValue,wordPos,abbPos)
        WORD.setRootValue(rootValue)
        wordsInSentence1.append(WORD)
        #WORD.printWord()
        verbResult1 = verbResult1+str(WORD.getRootValue())+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(verbWordId1)+str(tag)+str(1)+" "
        #nounResult1 = nounResult1+str(key)+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId1)+str(tag)+str(1)+" "
        verbWordId1+=1
    #print verbResult1
    # sentence1 contains all the important verb and nouns from sentence1, all as string for to be used by UKB 
    sentence1 = nounResult1+verbResult1
    #print sentence1
    
    file.write(sentence1)
    
    # this is a list of Word objects.
    wordsInSentence2 = []
    file.write("\n")
    file.write("ctx_02")
    file.write("\n")
    # noun phrase of sentence 2
    for key,value in nounDic2.items():
        #print key, value
        wordValue = key
        abbPos = value
        if(value[0]=='N'):
            wordPos = "NOUN"
            rootValue = wn.morphy(key,wn.NOUN)
            if(rootValue is None):
                rootValue = key
        WORD = Word(wordValue,wordPos,abbPos)
        WORD.setRootValue(rootValue)
        wordsInSentence2.append(WORD)
        #WORD.printWord()
        nounResult2 = nounResult2+str(WORD.getRootValue())+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId2)+str(tag)+str(1)+" "
        #nounResult1 = nounResult1+str(key)+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId1)+str(tag)+str(1)+" "
        nounWordId2+=1
    #print nounResult2
    
    # verb phrase of sentence 1
    for key,value in verbDic2.items():
        #print key, value
        wordValue = key
        abbPos = value
        if(value[0]=="V"):
            wordPos = "VERB"
            rootValue = wn.morphy(key,wn.VERB)
        WORD = Word(wordValue,wordPos,abbPos)
        WORD.setRootValue(rootValue)
        wordsInSentence2.append(WORD)
        #WORD.printWord()
        verbResult2 = verbResult2+str(WORD.getRootValue())+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(verbWordId2)+str(tag)+str(1)+" "
        #nounResult1 = nounResult1+str(key)+str(tag)+str(value[0].lower())+str(tag)+str("w")+str(nounWordId1)+str(tag)+str(1)+" "
        verbWordId2+=1
    #print verbResult2
    # sentence1 contains all the important verb and nouns from sentence1, 
    sentence2 = nounResult2+verbResult2
    #print sentence2    
    file.write(sentence2)
    file.close()
    # the file is saved and next is run the ukb to get the sense numbers back for ever word
    #print len(wordsInSentence1)
    #print len(wordsInSentence2)
    '''
    for w in wordsInSentence1:
        print w.printWord()
    print "--------------"
    for w in wordsInSentence2:
        print w.printWord()
    '''
    extractSense(filename) 
    #print "*************************Oneway inside readWSD**************************"
    words1 = readWSD("sensedcontext.txt",wordsInSentence1,wordsInSentence2)
    
    #for w in words1:
    #    w.printWord()
    #print "*************************the other way****************************"
    words2 = readWSD("sensedcontext.txt",wordsInSentence2,wordsInSentence1)
    
    #for w in words2:
    #    w.printWord()
    return words1, words2
    


if __name__=='__main__':
    '''
    1. Read the simple text file and extract the paragraphs in a list"
    2. Read the normal text file and extract the paragraphs in a list"
    '''
    simpleParas = extractParasInList('simpleCarnivore.txt')
    normalParas = extractParasInList('normalCarnivore.txt')
    
    #3. remove subtitles from the list of simple and normal paragraphs"
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
    
    #call it one time only, I comment the writing line, remember!!!
    #compute_tfidf(normalParas,"normalCarnivoretfidf.txt")
    #compute_tfidf(simpleParas,"simpleCarnivoretfidf.txt")
    
    #allSimpleWords,allNormalWords = extractAllWordsFromPara("simpleCarnivoretfidf.txt","normalCarnivoretfidf.txt")
    topSimpWords,topNormalWords = extractTopWordsFromPara("simpleCarnivoretfidf.txt","normalCarnivoretfidf.txt") # based on paragraphs
    #matchingPara(allSimpleWords,allNormalWords)
    matchingPara(topSimpWords,topNormalWords)
    print pairedPara  
    print "THIS IS COMBINED---> on Carnivore"
    '''
    find the words and its weight
    '''
    alignText(orgSimpleParas,orgNormalParas, pairedPara)