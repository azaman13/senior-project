#!/usr/bin/python
import copy
class Word(object):
    synSet="null"
    rootValue=""
    senseNo = -1
    weight = 0
    def __init__(self, value,pos,abbPos):
        self.value = value
        self.pos = pos
        self.abbPos = abbPos
    
    def setSynSet(self,s):
        self.synSet = s
    def setWeight(self,w):
        self.weight = w
    def setSenseNo(self,s):
        self.senseNo = s
    def setValue (self, v):
        self.value = v
    def setPos(self, p):
        self.pos = p
    def setAbbPos(self, abpos):
        self.abbPos = abpos
    def setRootValue(self,r):
        self.rootValue = r

    def getSynSet(self):
        return self.synSet
    def getWeight(self):
        return self.weight
    def getSenseNo(self):
        return self.senseNo
    def getValue (self):
        return self.value 
    def getPos(self):
        return self.pos
    def getAbbPos(self):
        return self.abbPos
    def getRootValue(self):
        return self.rootValue

    def copy(self):
        return copy.deepcopy(self)
            
    def printWord(self):
        print "Word:"+ str(self.getValue())+"\t",
        print "POS:"+str(self.getPos())+"\t",
        print "Abbreviated POS:"+str(self.getAbbPos())+"\t",
        print "SenseNo:"+str(self.getSenseNo())+"\t",
        print "Root:"+ str(self.getRootValue())+"\t",
        print "synset: "+ str(self.getSynSet())+"\t",
        print "Weight:"+str(self.getWeight())
