#!/usr/bin/python

__author__ = 'Roman Sharykin rs4da'

import sys
import os
import numpy as np
import math
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB


def transfer(fileDj, vocabulary):
	wordnet_lemmatizer = WordNetLemmatizer()
	raw = fileDj.read()
	BOWDj = [0]*(len(vocabulary)+1)
	for token in raw.split( ):
		token_stemed = wordnet_lemmatizer.lemmatize(token)
		if token_stemed in vocabulary:
			index = vocabulary.index(token_stemed) + 1
		else:
			index = 0
		BOWDj[index] = BOWDj[index] + 1
	return BOWDj


def loadData(Path):
	train_path = Path + '/training_set'
	test_path = Path + '/test_set'
			
	
	vocab_size = 100
	vocab_freq_threshold = 3

	word_freq = {}
	wordnet_lemmatizer = WordNetLemmatizer()
	for file in os.listdir( train_path+'/neg' ):
		f= open( train_path+'/neg' + '/' + file)
		raw = f.read()
		for word in raw.split( ):
			word_stemed = wordnet_lemmatizer.lemmatize(word)
			if word_stemed in word_freq:
				word_freq[word_stemed] = word_freq[word_stemed] + 1
			else:
				word_freq[word_stemed] = 0

	for file in os.listdir( train_path+'/pos' ):
		f= open( train_path+'/pos' + '/' + file)
		raw = f.read()
		for word in raw.split( ):
			word_stemed = wordnet_lemmatizer.lemmatize(word)
			if word_stemed in word_freq:
				word_freq[word_stemed] = word_freq[word_stemed] + 1
			else:
				word_freq[word_stemed] = 0
	global dict_predefined
	dict_predefined = sorted(word_freq, key=word_freq.get, reverse=True)[:vocab_size]
	dict_predefined = []
	for w, freq in word_freq.items():
		if(freq >= vocab_freq_threshold):
			dict_predefined.append(w)

	Xtrain = []
	ytrain = []
	Xtest = []
	ytest = []

	for file in os.listdir( train_path+'/neg' ):
		f= open( train_path+'/neg' + '/' + file)
		fearture_vector = transfer(f, dict_predefined)
		Xtrain.append(fearture_vector)
		ytrain.append(-1)

	for file in os.listdir( train_path+'/pos' ):
		f= open( train_path+'/pos' + '/' + file)
		fearture_vector = transfer(f, dict_predefined)
		Xtrain.append(fearture_vector)
		ytrain.append(1)

	for file in os.listdir( test_path+'/pos' ):
		f= open( test_path+'/pos' + '/' + file)
		fearture_vector = transfer(f, dict_predefined)
		Xtest.append(fearture_vector)
		ytest.append(1)

	for file in os.listdir( test_path+'/neg' ):
		f= open( test_path+'/neg' + '/' + file)
		fearture_vector = transfer(f, dict_predefined)
		Xtest.append(fearture_vector)
		ytest.append(-1)

	return Xtrain, Xtest, ytrain, ytest


def naiveBayesMulFeature_train(Xtrain, ytrain):

	thetaPos = []
	thetaNeg = []
	total_words_pos = 0
	total_words_neg = 0
	dict_count_pos = [0]*len(Xtrain[0])
	dict_count_neg = [0]*len(Xtrain[0])
	for a, label in zip(Xtrain, ytrain):
		for j in range(len(a)):
			if(label == 1):
				total_words_pos = total_words_pos + a[j]
				dict_count_pos[j] = dict_count_pos[j] + a[j]
			elif(label == -1):
				total_words_neg = total_words_neg + a[j]
				dict_count_neg[j] = dict_count_neg[j] + a[j]
	for i in range(len(Xtrain[0])):
		thetaPos.append((dict_count_pos[i] + 1.0)/(total_words_pos + len(Xtrain[0])) * 1.0)
		thetaNeg.append((dict_count_neg[i] + 1.0)/(total_words_neg + len(Xtrain[0])) * 1.0)
				
	return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):
	yPredict = []
	accurate_count = 0
	P_Pos = math.log(0.5)
	P_Neg = math.log(0.5)
	for a, label in zip(Xtest,ytest):
		P_Pos_a = P_Pos
		P_Neg_a = P_Neg
		for i in range(len(Xtest[0])):
			P_Pos_a = P_Pos_a + math.log(thetaPos[i]) * a[i]
			P_Neg_a = P_Neg_a + math.log(thetaNeg[i]) * a[i]
		if(P_Pos_a > P_Neg_a):
			yPredict.append(1)
			if(label == 1): accurate_count = accurate_count + 1
		else:
			yPredict.append(-1)
			if(label == -1): accurate_count = accurate_count + 1
	Accuracy = accurate_count/len(Xtest) * 1.0
	return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
	clf = MultinomialNB()
	clf.fit(Xtrain, ytrain)
	Accuracy = clf.score(Xtest,ytest)
	return Accuracy

def naiveBayesBernFeature_train(Xtrain, ytrain):
	thetaNegTrue = [0] * len(Xtrain[0])
	thetaPosTrue = [0] * len(Xtrain[0])
	for i in range(len(Xtrain[0])):
		count = 1
		for j in range ( int(len(Xtrain)/2) ):
			if(Xtrain[j][i] != 0):
				count = count + 1
		thetaNegTrue[i] = float(count/(len(Xtrain)/2 + 2))
		count = 1
		for j in range ( int(len(Xtrain)/2), len(Xtrain)):
			if(Xtrain[j][i] != 0):
				count = count + 1
		thetaPosTrue[i] = float(count/(len(Xtrain)/2 + 2))

	return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
	yPredict = []
	accurate_count = 0
	for i in range(len(Xtest)):
		pos_score = 0
		neg_score = 0
		
		for j in range(len(Xtest[i])):
			if(Xtest[i][j] == 0 ):
				pos_score = pos_score + math.log(1-thetaPosTrue[j])
				neg_score = neg_score + math.log(1-thetaNegTrue[j])
			else:
				pos_score = pos_score + math.log(thetaPosTrue[j])
				neg_score = neg_score + math.log(thetaNegTrue[j])
					
		if(pos_score >neg_score):
			yPredict.append(1)
			if(ytest[i] == 1):
				accurate_count = accurate_count+1
			else:
				yPredict.append(-1)
				if(ytest[i] == -1):
					accurate_count = accurate_count+1
				
	Accuracy = float(accurate_count/len(ytest))
	return yPredict, Accuracy

def output_theta(theta, pos):
	if pos == True:
		with open('theataPos.txt', 'w+') as f:
			for pred in theta:
				f.write(str(pred) + '\n')
	else:
		with open('theataNeg.txt', 'w+') as f:
			for pred in theta:
				f.write(str(pred) + '\n')

def output_thetaBern(theta, pos):
	if pos == True:
		with open('theataPosTrue.txt', 'w+') as f:
			for pred in theta:
				f.write(str(pred) + '\n')
	else:
		with open('theataNegTrue.txt', 'w+') as f:
			for pred in theta:
				f.write(str(pred) + '\n')

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print ("Usage: python naiveBayes.py dataSetPath testSetPath")
		sys.exit()

	print ("--------------------")
	textDataSetsDirectoryFullPath = sys.argv[1]
	testFileDirectoryFullPath = sys.argv[2]


	Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


	thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
	output_theta(thetaPos, True)
	output_theta(thetaNeg, False)

	yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
	print ("MNBC accuracy =", Accuracy)

	Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
	print ("MultinomialNB accuracy =", Accuracy_sk)

	thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
	output_thetaBern(thetaPos, True)
	output_thetaBern(thetaNeg, False)

	yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
	print ("BNBC accuracy =", Accuracy)
	print ("--------------------")
