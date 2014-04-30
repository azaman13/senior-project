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
import sys
import sets
import math
import itertools
import os
#shared
PAIREDSENTENCES=open("alignedSentencePairs.txt","w+")
pairedPara = {}
simpleParas = []
normalParas = []
orgSimpleParas = []
orgNormalParas = []
# shared things  end
v1 = {}
v2 = {}
n1 = {}
n2 = {}
PUNCTLIST = [',','.',';','?',':','!','(',')','{','}','/','==','-','"']
STOPWORDS = stopwords.words('english')
TAGLIST = ['CC','CD' ,'DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ', 'WDT','WP','WRB']
commonAuxilaryVerbs  = ['is', 'are', 'was', 'were', 'am', 'be', 'been', 'will', 'shall', 'have', 'has', 'had', 'would', 'could', 'should', 'do', 'does', 'did', 'can', 'may','might', 'must', 'seem']
''' 
    @Description: This method sorts a dictionary of word, tf_idf weight (key,value) pairs
    @param: a dictionary
    @Return: a sorted dictionary
'''
def sortDict(dict):
    items = [(v, k) for k, v in dict.items()]
    items.sort()    
    items.reverse()
    dict = [(k, v) for v, k in items]
    #for key,value in dict:
        #print "%s: %s" % (key, value)
    return dict
    
# this method is not in semantic version only
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
    @Description: This method returns all the paragraphs of the given file into a list
    
    @param: name of the text file which needs to be parsed, and return a list of parapraphs
    
    @return: a list whose every item is a paragraph (of  type string)
'''
def extractParasInList(name):

    corpuslocation ='/home/aniszaman/seniorProject/semantic/august/'
    
    reader = PlaintextCorpusReader(corpuslocation, '.*\.txt')
    # This gives the list of paragraphs. every paragraph list contains ist of sentences
    # So it is a list of lists. Bunch of sentenses as a list joins together to make         #lists of pararagraph
    pList = []
    paragraphlist =  reader.paras(name) #'simpleAugust.txt'
    numpara = len(paragraphlist)
    for sentlist in paragraphlist:
        #print sentlist
        numsent = len(sentlist)
        paraAsAList = []
        # this loops through all the sentence lists and make them one list'''
        for i in range(numsent):
                paraAsAList = paraAsAList + sentlist[i] 
        paraAsAString = ""
        for word in paraAsAList:
                paraAsAString = paraAsAString + word + str(" ")
        #print paraAsAString
        pList.append(paraAsAString)
        #print len(pList)
    return pList


'''
    @Description:Given a text file (having different paragraphs seperated by "\n", 
    this method generates the tfidf ') for every word in that paragraph and 
    saves them to a file.
    
    @params: text--> list of paragraphs
         filename--> the target file name where the tf-idf should be saved
    
    @return: Nothing, just saves a file in the current directory
'''
def compute_tfidf(text,filename):
    numPara = len(text)
    #print "there should be this many para in the text file ", numPara
    
    colList = []
    paragraphWords = []
    for i in range(numPara):
        paragraphWords = word_tokenize(text[i])
        colList.append(paragraphWords)
    
    collection = TextCollection(colList)
     
    for paraList in colList:
        dict={}
        for term in paraList:
            #print term, "has weight: ", collection.tf_idf(term,paraList)
            dict[term]= collection.tf_idf(term,paraList)
        
        d=sortDict(dict)

        textFile=open(filename,"a")
        textFile.write("\n")

        for key,value in d:
            s = str(key) + "\t" + str(value)+"\n"
            #print s
            textFile.write(s)
        #print dict
    

'''
@Description: Removes == Refereance == types things from the article
@param: a list
@return: a new list where paragraphs starting with "==" are removed
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
'''
    @Description: Given the two text files, extract all words from each paragraphs
    @Params: simpleFileName --> simple textfile's tf-df values
             normalFileName --> normal textfile's tf-idf values
    @return: two list (one for simple file and other for normal file) with 
             all the words with their tf-idf scores
'''

def extractAllWordsFromPara(simpleFileName,normalFileName):
    simpleParagraphsList = []
    normalParagraphsList = []
    numPara=0
    textFile=open(simpleFileName,"r")
    for line in textFile.readlines():
        itemList = word_tokenize(line)
        simpleParagraphsList.append(itemList)     
    # this is a list of paras with ALL tfidf words selected. format is [ [P1] , [P2]  [P3] ... [Pn]    ], where Pn is  a (2 element) 
    # list of all words
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
'''
    *****************ASSUMING EVERY PARAGRAPHS HAS ATLEAST 15 words****************
    @Description: Given the two text files, extract the top 15 words from each paragraphs
    @Params: simpleFileName --> simple textfile's tf-df values
             normalFileName --> normal textfile's tf-idf values
    @return: two list (one for simple file and other for normal file) with the 
             the top words with their tf-idf scores
'''
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
    @Description: This method computes dot product of two given lists of equal length
    @Params: list1, list2 are just simple list of floating numbers
    @return: the dot product between the two input list
'''

def computeDotProduct(list1,list2):
    result = 0.0
    for i in range(len(list1)):
        result = result + (float(list1[i]) * float(list2[i]))
    return result


'''
    @Description: Given Two Paragraphs, this returns the list of common shared words. 
    @Params: para1 and  para2 are lists of this form [ ...[   ,  ] , [   ,   ], [   ,   ] ......      ] 
    @return: List of shared word and the dot product
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
    @Description: Given two list of paragraphs, this method find which paragraph 
    from one (simple) article matches with which paragraph from the normal article
    @param: simpleDoc--> List of paragraphs
            normalDoc--> list of paragraphs
    @return: Nothing, but it updates the Global pairedPara dictionary that holds the 
    the mapping information between the paragraphs of two articles

    Note: This method calls findCommonWords() which in turn calls computeDotProduct()
'''

def matchingPara(simpleDoc,normalDoc):
    # Calling findCommonWords method to find the common words between the paragraphs"
    # findCommonWords is Calling computeDotProduct=> to find the similarity between the shared words "
    #print len(simpleDoc)
    big = []
    for sPara in simpleDoc: # format of sPara => [ [word, #],[word,#].... ]
        commonWord = []
        maxWeight = 0
        whichNormalPara=0
        lst = []
        
        for nPara in normalDoc: #format of nPara => [ [word, #],[word,#].... ]
            #print "These are the shared words between the simple and normal",simpleDoc.index(sPara),normalDoc.index(nPara)
            #j = 0
            commonWord, dotProduct = findCommonWords(sPara,nPara)
            #print "The common Words are =====================================\n"
            #print commonWord, dotProduct

            if(dotProduct>maxWeight):
                maxWeight = dotProduct
                whichNormalPara = normalDoc.index(nPara)
                lst = commonWord

        pairedPara[simpleDoc.index(sPara)] = whichNormalPara

'''
    @Description: Given two paragraphs, this method returns two list containing 
                  the simple sentence and normal sentences

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

'''This method us used for tf_idf part'''

def sentenceAlignment(simpleParas, normalParas, pairedPara): 
    #print simpleParas, len(simpleParas)
    #print normalParas, len(normalParas)
    for key,value in pairedPara.items(): # key is simple and value in normal
        print "======================================================="
        print "======================================================="
        print "======================================================="
        print "======================================================="
        
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
            #print "<><><><<><><><><><><<><><<><>><><"
            #print "similarity=",maxSimilarity
            #print simpleSentence
            #print normalSentence
        ## END OF ONE PAIRED PARA

'''
    @Description: If a sentence is not parsable by Stanford parser, i.e after attempting to parse, 
    the parser wrote the following in the parsed file: 
                Sentence skipped: no PCFG fallback.\n
                SENTENCE_SKIPPED_OR_UNPARSABLE\n'
    @param: filename--> the file to be verified
    #return: boolean 
'''
def verifyParsedFile(filename):
    textFile=open(filename,"r")
    numLines = textFile.readlines()
    if len(numLines)==2:
        if numLines[1] == 'SENTENCE_SKIPPED_OR_UNPARSABLE\n':
            return False 
        return True

'''
pairedPara = dic of which para match with which one
simpleParas, normalParas are dic of paragraphs (key, value) => (para number, paragraph)
'''
def alignText(simpleParas, normalParas, pairedPara): 
    #print simpleParas, len(simpleParas)
    #print normalParas, len(normalParas)
    
    # "inside align text"
    for key,value in pairedPara.items(): # key is simple and value in normal   
        SPara = simpleParas[key]
        #print SPara
        #print value, len(normalParas), len(simpleParas),pairedPara
        NPara = normalParas[value]
        print "======================================================="
        print "======================================================="
        print "======================================================="
        print "======================================================="

        simpleSentenceList = []
        normalSentenceList = []
        # here I have the sentences stored in a list.
        # Now I should take every pair, save to a file, and then 
        # do the sementic part
        simpleSentenceList,normalSentenceList = formSentenceListForSementic(SPara,NPara)
        #print "!@#$%^&*!@#$%^&*@#$%^"
        #print simpleSentenceList
        #print normalSentenceList
        #print "!@#$%^&*()!@#$%^&*("
        # normalSentenceList = simpleSentenceList = [s1,s2,s3...]
        '''ANIS HERE YOU HAVE TO remove some jargon from the sentence'''
        for sentence1 in simpleSentenceList:
            #write sentence1 to a file, parse it and then verify it
            simplefilename = "sentence1.txt"
            SFile=open(simplefilename,"w+")
            SFile.write(sentence1)

            SFile.close()
            parseFile("sentence1.txt")
            # if failed to parse, skip this sentence and continue
            if verifyParsedFile("parsedsentence1.txt")  == False:

                continue
            for sentence2 in normalSentenceList:
                
                normalfilename = "sentence2.txt"
                NFile=open(normalfilename,"w+")
                NFile.write(sentence2)
                NFile.close()
                parseFile("sentence2.txt")
                #check whether parsing was done properly
                # if failed to parse, skip this sentence and continue
                if verifyParsedFile("parsedsentence2.txt")  == False:
                    continue
                #print  "calling buildClause method"
                #print "=======SENTENCE ONE======="
                buildClause("parsedsentence1.txt", "one")
                #print "=======SENTENCE Two======="
                buildClause("parsedsentence2.txt","two")

                #print "calling makeContextFile method"
                print "<><><><><><><><><><><><><><><><><><><><><><><><><><><>"
                print sentence1
                print "------------------------------------------------------"
                print sentence2
                #PAIREDSENTENCES.write(sentence1)
                #PAIREDSENTENCES.write("\n")
                #PAIREDSENTENCES.write(sentence2)
                #PAIREDSENTENCES.write("\n")
                makeContextFile(n1,v1,n2,v2)

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
'''
    @param: simpleLine and normalLine are two lists
    @return: union of the two lists
'''
def union(simpleLine,normalLine):
    unionSet = []
    simpleSet = sets.Set(simpleLine)
    normalSet = sets.Set(normalLine)
    temp = simpleSet.union(normalSet)
    unionSet = list(temp)
    return unionSet
'''
    @Description: This parses the file specified and saves the parse tree
                in a file
    @param: filename-> file to be parsed
    @return: none, creates a parse tree of the input sentence
'''
def parseFile(fileName):
    filePath  = "/home/aniszaman/seniorProject/semantic/august/"+fileName
    name = "parsed"+fileName
    outputFile=open(name,"w+")
    outputFile.close()
    cmd = '/home/aniszaman/seniorProject/stanford-parser/lexparser.sh '+filePath+' >'+name
    #print cmd
    #print filePath 
    os.system(cmd)
    
'''
    @Description: find similarity betwwn word senses of two words
    @Params: worda, wordb are two words we are intersted
    @return: none
    THIS IS A HELPER METHOD TO TEST DIFFERENT THINGS AT DIFFERENT TIMES
'''
def getSenseSimilarity(worda,wordb):
    """
    find similarity betwwn word senses of two words
    """
    wordasynsets = wn.synsets(worda)
    wordbsynsets = wn.synsets(wordb)

    synsetnamea = [wn.synset(str(syns.name)) for syns in wordasynsets]
    synsetnameb = [wn.synset(str(syns.name)) for syns in wordbsynsets]

    for sseta, ssetb in [(sseta,ssetb) for sseta in synsetnamea for ssetb in synsetnameb]:

        pathsim = sseta.path_similarity(ssetb)

        wupsim = sseta.wup_similarity(ssetb)

        if pathsim != None:

            print "Path Sim Score: ",pathsim," WUP Sim Score: ",wupsim, "\t",sseta.definition, "\t", ssetb.definition

'''
    @Description: This method is little complicated because it reads the parse tree
    and extract the noun clause and get the principle verbs and nouns from 
    the clauses. It does it for both the sentences.
    @Params: filenamne---> name of the file that contain the parse tree
             whichSentence---> which sentence we are intersted in
    @return: none, but updates the global n1,v1,n2,v2 dictionaries
'''
def buildClause(filename,whichSentence):

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
        if(len(LIST)>0): #------------------------------------->>>> I added this if cause the LIST was getting empty
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
    

    NOUN=[]
    VERB=[]
                
    for i in range(len(MAP)):
        
        if(i==0):
            #print "NOUN"
            NOUN = LIST[i:MAP[i]+1]
            #print NOUN
        
        if(i==len(MAP)-1):
   
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
    @Description: This method will run the ukb personalized page ranking to get 
                  the sense of every word
    @params: the name of the context file. filename -> name of file where the important nouns and verbs of the simple and normal sentence exist
    @return: none, creates a file that contain the database location of the words from the input file
'''
def extractSense(fileName):
    binFile = "/home/aniszaman/seniorProject/ukb-2.0/bin/wn30.bin"
    dicFile = "/home/aniszaman/seniorProject/ukb-2.0/lkb_sources/30/wnet30_dict.txt"
    # finally this will change
    inputFile = "/home/aniszaman/seniorProject/semantic/august/"+fileName
    ukbPath  = "/home/aniszaman/seniorProject/ukb-2.0/bin/ukb_wsd"
    wordSenseFileName = "sensed"+fileName
    # just creating the file, otherwise, the result gets concatenated 
    f=open(wordSenseFileName,"w+")
    f.close()

    command = ukbPath+  " --ppr -K "+binFile+" -D "+dicFile+" "+inputFile+ '>' + wordSenseFileName
    os.system(command)
'''
    @DESCRIPTION: This is a helper method that removes all word object whose weight is 0
    @Param: allwords--> List of Word objects
    @return: a list where no word has weight 0
'''
def filterWords(allWords):
    for word in allWords:
        if(word.getWeight()<0):
            allWords.remove(word)
    return allWords

'''
    @Description: Read the sense file and update the wsd of every word 
                  of the  sentence1 & sentence2 and compute the similarity"
    @Param: sentence1 & sentence2 -> list of Word objects.
    @Return: none
'''
def readWSD(wsdFile, sentence1, sentence2):
    
    textFile=open(wsdFile,"r")
    lines = []
    for line in textFile.readlines():
        lines.append(line)
    del lines[0]
    #lines look like this: ['ctx_01 w2  02684924-v !! continue\n',.... 'ctx_02 w1  01056411-n !! stop\n', ]
    
    for l in lines:
        r = l.split()
        if(r[0]=='ctx_01'):
            # will work on sentence1
            root = r[len(r)-1]
            senseVal = r[len(r)-3]
            #print root, senseVal
            #print type(senseVal)
            # this loop gets rid of any precidding 0 from the database location/ synset location
            while(senseVal[0]=="0"):
                senseVal = senseVal[1:]
         
            for w in sentence1:
                if(w.getRootValue()==root and isinstance("",type(senseVal))):
                    # splitting '13244109-n'  ---> ['13244109', 'n']
                    senseVal = senseVal.split('-')
                    p = senseVal[len(senseVal)-1]
                    num = int(senseVal[0])
                    #print p, num, w.value
                
                    if(num!=-1):
                        s = wn._synset_from_pos_and_offset(p, num)
                        
                        s = s.__str__()
                        s = s.split("'")[1]
                        #print s
                        #s = s.split('.')
                        #s = s[len(s)-1]
                        #senseId = s[:len(s)-2]
                        #print senseId, w.value
                        #print s.__str__()
                        #w.setSenseNo(str(senseId))  ################### convert the senseVal to a nice number            
                        w.setSynSet(s)
                    #print "sentence1"
    
        
        if(r[0]=='ctx_02'):
            root = r[len(r)-1]
            senseVal = r[len(r)-3]
            #print root, senseVal
            # this loop gets rid of any precidding 0 from the database location/ synset location
            while(senseVal[0]=="0"):
                senseVal = senseVal[1:]

            for w in sentence2:
                if(w.getRootValue()==root and isinstance("",type(senseVal))):
                    senseVal = senseVal.split('-')
                    p = senseVal[len(senseVal)-1]
                    num = int(senseVal[0])
                    
                    #print w.getRootValue(), p, num, w.value
                    if(num!=-1):
                        s = wn._synset_from_pos_and_offset(p, num)
                        s = s.__str__()
                        s = s.split("'")[1]
                        w.setSynSet(s)
    #print "============="
    #for w in sentence1:
    #    w.printWord()
    #print "-----------"
    #for w in sentence2:
    #    w.printWord()
    
    # At this point I have the words from sentence 1 and sentence 2 in perfect way, 
        
    # REMEMBER: HERE I NEED TO FIND WAY TO CONVERT THE NUMBERS to nicer sense values 
    SENT1NOUNS = []
    SENT1VERBS = []
    SENT2NOUNS = []
    SENT2VERBS = []

    for w1 in sentence1:
        if(w1.getPos()=='NOUN'):
            SENT1NOUNS.append(w1)
        if(w1.getPos()=='VERB'):
            SENT1VERBS.append(w1)
        #w1.printWord()
    
    for w2 in sentence2:
        if(w2.getPos()=='NOUN'):
            SENT2NOUNS.append(w2)
        if(w2.getPos()=='VERB'):
            SENT2VERBS.append(w2)
        #w2.printWord()
    
    # Finding the weights for nouns
    # record the max similarity values, later just add them
    maxNounSimilarityValues = []
    # this is for me to see which word from sentence 1 maps to the corresponding words in sentence 2
    nounMapping = {}

    for n1 in SENT1NOUNS:
        if(n1.getSynSet()!="null"):
            #n1.printWord()
            exp1 = n1.getSynSet()
            noun1 = wn.synset(exp1)

            maxValue = 0
            # just for printing
            ntemp = ""
            for n2 in SENT2NOUNS:
                if(n1.getSynSet()!="null"):
                    exp2 = n2.getSynSet()

                    #print exp2, type(exp2)
                    if(exp2!='null'):
                        noun2 = wn.synset(exp2)
                        value = noun1.path_similarity(noun2)

                        if(value > maxValue):
                            maxValue = value
                            maxNounSimilarityValues.append(maxValue)
                            nounMapping[n1.getValue()] = n2.getValue()
                            ntemp = n2.getValue()
            #print "the max similarity between " + n1.getValue() + " and "+ntemp +"is" + str(maxValue)
    # record the max similarity values, later just add them
    maxVerbSimilarityValues = []
    # this is for me to see which word from sentence 1 maps to the corresponding words in sentence 2
    verbMapping = {}
    
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

                    if(value > maxValue):
                        maxValue = value
                        maxVerbSimilarityValues.append(maxValue)
                        verbMapping[v1.getValue()] = v2.getValue()
                        vtemp = v2.getValue()

            #print "the max similarity between " + v1.getValue() + " and "+vtemp +"is" + str(maxValue)

    similarityValue = 0
    for v in maxVerbSimilarityValues:
        similarityValue = similarityValue+v
    for n in maxNounSimilarityValues:
        similarityValue = similarityValue+n
    print "similarity-------> ",similarityValue

'''
makeContextFile -> this takes in the dic of Noun and Verbs (word:pos format) of two sentences and generate the 
                    input file that is used to get the sense of every word.
    it calls the extractSense method, runs the ukb ppr and save the content in a file  
'''
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
    # sentence1 contains all the important verb and nouns from sentence1, 
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
    
    extractSense(filename) 
    readWSD("sensedcontext.txt",wordsInSentence1,wordsInSentence2)

if __name__=='__main__':
    '''
    1. Read the simple text file and extract the paragraphs in a list"
    2. Read the normal text file and extract the paragraphs in a list"
    '''
    simpleParas = extractParasInList('simpleAugust.txt')
    normalParas = extractParasInList('normalAugust.txt')
    ###for debugging###
    #print "simple article has " + str(len(simpleParas)) + " paragraphs"
    #print "normal article has " + str(len(normalParas)) + " paragraphs"
   
    
    #3. remove subtitles from the list of simple and normal paragraphs"
    simpleParas = removeSubtitles(simpleParas)
    normalParas = removeSubtitles(normalParas)
    
    # So orgSimpleParas/ orgNormalParas = [P1,P2,P3,...] list of paragraphs
    orgSimpleParas = simpleParas
    orgNormalParas = normalParas
    
    # remove stopwords and punctuations from simple paragraphs"
    SP = [] # nicely formatted simple Paragraphs list
    for para in simpleParas:
        words = word_tokenize(para)
        # Removing any stop words and punctuations from a particular paragraphs
        for term in words:
            if (term in STOPWORDS or term in PUNCTLIST):
                words.remove(term)        
        # After the unnecessary stop words and punctuations are removed, reconstruct the paragraph
        temp = " "
        for word in words:
            temp = temp + str(word)+str(" ")
        SP.append(temp)

    simpleParas = SP
   
    # remove stopwords and punctuations from normal paragraphs"
    NP = [] # nicely formatted simple Paragraphs list
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

    # Compute tf-idf one time only. It is a good idea to save some 
    # computation power
    # computing the tf_idf and saving them to files"
    #compute_tfidf(normalParas,"normalAugusttfidf.txt")
    #compute_tfidf(simpleParas,"simpleAugusttfidf.txt")
    
    # get all/top words from the paragraphs as allSimpleWords,allNormalWords by calling extractAllWordsFromPara method"
    #allSimpleWords,allNormalWords = extractAllWordsFromPara("simpleAugusttfidf.txt","normalAugusttfidf.txt")
    topSimpWords,topNormalWords = extractTopWordsFromPara("simpleAugusttfidf.txt","normalAugusttfidf.txt") # based on paragraphs
    
    # calling matching method -> return the mapping of paragraphs"
    matchingPara(topSimpWords,topNormalWords)
    print pairedPara
        

    # This is the new part
    #print "11. alignText: matches respective paragraphs"
    #print "12. (still inside alighText) then for every pair of paragraphs, it calls formSentenceListForSementic"
    #print "13. parse: Parse the respective sentneces"
    #print "14. verifyParsedFile: Verify that the parsing was legitimate"
    #print "15: buildClause: from the parse tree, build the verb and noun clauses"
    #print "16: makeContextFile : this takes in the dic of Noun and Verbs (word:pos format) of two sentences and generate the input file that is used to get the sense of every word. it calls the extractSense method"
    #print "17. extractSense: This method will run the ukb personalized page ranking to get the sense of every word"
    #print "18. readWSD: read the sense file and update the wsd of every word of the sentence1 & sentence2 and compute the similarity"


    alignText(orgSimpleParas,orgNormalParas, pairedPara)
    