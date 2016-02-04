# William Burke
# Naive Bayes
# Febuary 1st, 2016 


# imports
import sys
import numpy as np
import time
import matplotlib.pyplot as plt


# Used to tell me how long different parts of my code took to run
def times():
	global starttime
	endtime = time.time()
	total_time = endtime - starttime

	if total_time > 60:
		minutes = total_time / 60
		seconds = int(total_time % 60)
		print "Time To Run: ", minutes, "minutes and", seconds,"seconds"
	else:
		print "Time To Run: ", ("%.2f" % round(total_time,3))," seconds"

	#global starttime 
	starttime = time.time()


# Load files and count how many of each attribute for each classification
def Load_N_Count_Files(infile):
	
	above = np.zeros(124,dtype=int)
	below = np.zeros(124,dtype=int)
	aboveCount = 0
	belowCount = 0
	#Data = []

	Row = infile.readline() 
	i = 0
	while Row!="":
		variables = Row.split(" ")
		#Data.append(variables)
		if variables[0] == '-1':
			belowCount += 1
			variables.pop(0)

			for value in variables:
				if value =='\n':
					break
				num = value.replace(":1","")
				num = int(num)
				below[num] += 1
		if variables[0] == '+1':
			aboveCount += 1
			variables.pop(0)
			for value in variables:
				if value =='\n':
					break
				num = value.replace(":1","")
				num = int(num)
				above[num] += 1
		Row = infile.readline() 
		i = i + 1

	# Commands used to help debug:
	#print np.sum(above) + np.sum(below)
	# above = above/float(np.sum(above))
	# below = below/ float(np.sum(below))
	#print above
	#print below
	#print np.sum(above) + np.sum(below)
	return above, below

def Wheres_Ma_Bae(Counts_Above,Counts_Below, Attributes, Alpha):
	probAbove = 0
	SumAbove = np.sum(Counts_Above)
	SumBelow = np.sum(Counts_Below)

	for value in Attributes:
		if value =='\n':
			break
		value = value.replace(":1","") 
		value = int(value)

		count = Counts_Above[value]
		nextprob = np.log((count + Alpha)/float(SumAbove + Alpha*len(Counts_Above)))
		

		probAbove = probAbove - nextprob #(count + Alpha)/float(count + Alpha*len(Attributes))
		#print probAbove

	probBelow = 0
	for value in Attributes:
		if value =='\n':
			break
		value = value.replace(":1","") 
		value = int(value)

		count = Counts_Below[value]
		nextprob = np.log((count + Alpha)/float(SumBelow + Alpha*len(Counts_Below)))
		
		probBelow = probBelow - nextprob #(count + Alpha)/float(SumBelow + Alpha*len(Attributes))
		#print probBelow

	probAbove = probAbove - np.log((SumAbove/float(SumAbove+SumBelow)))
	probBelow = probBelow - np.log((SumBelow/float(SumAbove+SumBelow)))
	if probBelow < probAbove:
		return '-1'
	else: return '+1'



def Find_Alpha(above,below, infile):
	AlphaAccuracy = []
	for alpha in range(1,200):
		infile.seek(0)
		Row = infile.readline() 
		#print "here"
		correct = 0
		i = 0
		while Row!="":
			i = i + 1
			# if i ==5:
			# 	break
			variables = Row.split(" ")
			value = variables[0]
			variables.pop(0)
			guess = Wheres_Ma_Bae(above,below,variables,alpha)
			if value == guess:
				correct += 1
			Row = infile.readline()

		accuracy = correct/float(i)
		AlphaAccuracy.append(accuracy)
		#if accuracy > .5:
		print alpha,":", accuracy

	plt.plot(AlphaAccuracy)
	plt.title("Accuracy vs Alpha Value for Bayes Classifier of Adult Data Set")
	plt.xlabel("Alpha")
	plt.ylabel("Accuracy")
	plt.show()
	return (AlphaAccuracy.index(max(AlphaAccuracy))+1)

def Train_Accuracy(above, below, infile, alpha):
	infile.seek(0) # return to top of file
	Row = infile.readline() 
	#print "here"
	correct = 0
	i = 0
	while Row!="":
		i = i + 1
		# if i ==5:
		# 	break
		variables = Row.split(" ")
		value = variables[0]
		variables.pop(0)
		guess = Wheres_Ma_Bae(above,below,variables,alpha)
		if value == guess:
			correct += 1
		Row = infile.readline()

	accuracy = correct/float(i)
	
	print "The train set gave an accuracy of:"
	print correct, "correct predictions for",i,"points."
	print "The accuracy is", accuracy + "."
	


def Dev_Accuracy(above, below, infile, alpha):
	infile.seek(0)

	Row = infile.readline() 
	#print "here"
	correct = 0
	i = 0
	while Row!="":
		i = i + 1
		# if i ==5:
		# 	break
		variables = Row.split(" ")
		value = variables[0]
		variables.pop(0)
		guess = Wheres_Ma_Bae(above,below,variables,alpha) # find classification
		if value == guess:
			correct += 1
		Row = infile.readline()

	accuracy = correct/float(i)
	print "The dev set gave an accuracy of:"
	print correct, "correct predictions for",i,"points."
	print "The accuracy is", accuracy

def Test_Accuracy(above, below, infile, alpha):
	infile.seek(0)

	Row = infile.readline() 
	#print "here"
	correct = 0
	i = 0
	while Row!="":
		i = i + 1
		# if i ==5:
		# 	break
		variables = Row.split(" ")
		value = variables[0]
		variables.pop(0)
		guess = Wheres_Ma_Bae(above,below,variables,alpha)
		if value == guess:
			correct += 1
		Row = infile.readline()

	accuracy = correct/float(i)
	#print "The test set gave an accuracy of:"
	print ""
	print correct, "correct predictions for",i,"points."
	print "The accuracy is", accuracy
	print ""



def main():
	global starttime 
	starttime = time.time()

	train_file = 'a7a.train'
	Training_Data = open(train_file, 'r')
	#print len(Training_Data)

	above, below = Load_N_Count_Files(Training_Data)
	#times()
	

	dev_file = 'a7a.dev'
	Dev_Data = open(dev_file,'r')
	#times()

	#alpha = Find_Alpha(above, below, Dev_Data)
	alpha = 169
	#print "Alpha of highest accuracy:", alpha
	#times()
	#print ("")
	#Train_Accuracy(above, below, Training_Data,alpha)
	#print("")
	#Dev_Accuracy(above, below, Dev_Data,alpha)
	#print("")

	file_test =sys.argv[1]
	Testing_Data = open(file_test,'r')
	#times()
	

	Test_Accuracy(above, below, Testing_Data, alpha)
	#times()
main()



# def Wheres_Ma_Bae(Counts_Above,Counts_Below, Attributes, Alpha):
# 	probAbove = 1
# 	SumAbove = np.sum(Counts_Above)
# 	SumBelow = np.sum(Counts_Below)
# 	for value in Attributes:
# 		if value =='\n':
# 			break
# 		value = value.replace(":1","") 
# 		value = int(value)

# 		count = Counts_Above[value]
# 		nextprob = (count + Alpha)/float(SumAbove + Alpha*len(Attributes))
# 		if nextprob==0:
# 			nextprob = 0.00001
# 			print "Zeros"

# 		probAbove = probAbove* nextprob #(count + Alpha)/float(count + Alpha*len(Attributes))
# 		#print probAbove

# 	probBelow = 1
# 	for value in Attributes:
# 		if value =='\n':
# 			break
# 		value = value.replace(":1","") 
# 		value = int(value)

# 		count = Counts_Below[value]
# 		nextprob = (count + Alpha)/float(SumBelow + Alpha*len(Attributes))
# 		if nextprob==0:
# 			nextprob = 0.00001
# 		probBelow = probBelow*nextprob #(count + Alpha)/float(SumBelow + Alpha*len(Attributes))
# 		#print probBelow

# 	probAbove = probAbove * (SumAbove/float(SumAbove+SumBelow))
# 	probBelow = probBelow * (SumBelow/float(SumAbove+SumBelow))
# 	if probBelow > probAbove:
# 		return '-1'
# 	else: return '+1'

