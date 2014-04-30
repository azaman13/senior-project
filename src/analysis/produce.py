alg1NScore = []
alg1SScore = []
alg2NScore = []
alg2SScore = []

def main(filename, whichone):
	f = open(filename,"r")
	text = f.readlines()
	text = text[0]
	textList = text.split("},{")
	Scores = []

	for item in textList:
		number =  item.split(',')[1:][0]

		if(whichone=="alg1Org"):
			alg1NScore.append(number)

		if(whichone=="alg1Suff"):
			alg1SScore.append(number)


		if(whichone=="alg2Org"):
			alg2NScore.append(number)


		if(whichone=="alg2Suff"):
			alg2SScore.append(number)


if __name__ == '__main__':
	numBin = 25
	offset = .16
	LIST1 = []
	LIST2 = [] 
	LIST3 = []
	LIST4 = []
	for i in range(25):
		LIST1.append([])
		LIST2.append([])
		LIST3.append([])
		LIST4.append([])

	#print len(LIST1), LIST1

	main("alg1Org.txt","alg1Org")
	main("alg1Shuff.txt","alg1Suff")
	main("alg2NScore.txt","alg2Org")
	main("alg2SScore.txt","alg2Suff")

	alg1NScore[-1] = alg1NScore[-1][0:-1]
	alg1SScore[-1] = alg1SScore[-1][0:-1]
	#print alg2NScore
	alg2NScore[-1] = alg2NScore[-1][0:-1]
	alg2SScore[-1] = alg2SScore[-1][0:-1]
	
	for score in alg1NScore:
		binIndex = int(float(score)/(offset))
		#print score, "\t",binIndex
		if(binIndex==25):
			binIndex = binIndex-1
		LIST1[binIndex].append(score)

	for score in alg1SScore:
		binIndex = int(float(score)/(offset))
		#print score, "\t",binIndex
		if(binIndex==25):
			binIndex = binIndex-1
		LIST2[binIndex].append(score)		

	for score in alg2NScore:
		binIndex = int(float(score)/(offset))
		#print score, "\t",binIndex
		if(binIndex==25):
			binIndex = binIndex-1
		LIST3[binIndex].append(score)

	for score in alg2SScore:
		binIndex = int(float(score)/(offset))
		#print score, "\t",binIndex
		if(binIndex==25):
			binIndex = binIndex-1
		LIST4[binIndex].append(score)	
	'''

	#print LIST1
	start = 0
	end = start+offset
	print "-------------------------------"
	for bin in LIST1:
		#print LIST1.index(bin), "----->",start,end
		start = end
		end = end + offset
		#print len(bin),",",
		for i in bin:
			print i,",",

	start = 0
	end = start+offset
	print "\n -------------------------------"
	for bin in LIST2:
		#print LIST1.index(bin), "----->",start,end
		start = end
		end = end + offset
		#print len(bin),",",
		for i in bin:
			print i,",",
	
	num = 0
	start = 0
	end = start+offset
	print "\n -------------------------------"
	for bin in LIST3:
		#print LIST1.index(bin), "----->",start,end
		start = end
		end = end + offset
		#print len(bin),",",
		num = num+len(bin)
		for i in bin:
			print i,",",
	'''
	#print num		
	
	start = 0
	end = start+offset
	print "\n -------------------------------"
	for bin in LIST4:
		#print LIST1.index(bin), "----->",start,end
		start = end
		end = end + offset
		#print len(bin),",",
		#num = num+len(bin)

		for i in bin:
			print i,",",
	