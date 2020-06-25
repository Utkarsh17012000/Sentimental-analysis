#! /usr/bin/python3
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


replace_no_space = re.compile("[:.;!\'?,\"()\[\]]")
replace_with_space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def cleaning_func(reviews):
	reviews = [ replace_no_space.sub("",line.lower()) for line in reviews ]
	reviews = [ replace_with_space.sub(" ",line) for line in reviews ] 
	
	return reviews

reviews_test = open("full_test.txt","r")
reviews_train = open("full_train.txt","r")

reviews_train_clean = cleaning_func(reviews_train) 
reviews_test_clean = cleaning_func(reviews_test)

print("successful!!!")

print("\n\n")

cv = CountVectorizer(binary=True)#,stop_words='english', ngram_range=(2,2) for only bigrams )
cv.fit(reviews_train_clean)

#creating sparse matrix for test and training data
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

#building a classifier
target = [1 if i<12500 else 0 for i in range(25000)]

training_data, testing_data, training_values, testing_values = train_test_split(X, target, train_size = 0.75)
'''
for c in [0.01,0.05,0.25,0.5,1]:
	lr = LogisticRegression(C=c)
	lr.fit(training_data,training_values)
	print("Accuracy for C: {}, {}".format(c,accuracy_score(testing_values,lr.predict(testing_data))))
'''
#C=0.05 comes out to be the best fit

#finally train the model at penality of 0.05 on X_test
try:
	final_model = LogisticRegression(C=0.05)
	final_model.fit(X,target)
	predictions = final_model.predict(X_test)
	print("Accuracy(C=0.05): {}".format(accuracy_score(target,predictions)))
except:
	print("Error Found!")

print("\n\n")

#Sanity check
'''
print(len(final_model.coef_[0])) 
print(len(cv.get_feature_names())) #sparse matrix columns actually

feature_to_coef = {
	word:coef for word,coef in zip(
		cv.get_feature_names(), final_model.coef_[0]	
	)
}

for best_positive in sorted(
	feature_to_coef.items(),
	key = lambda x:x[1],
	reverse=True)[:5]:
	print(best_positive)

print("\n")

for best_negative in sorted(
	feature_to_coef.items(),
	key = lambda x:x[1])[:5]:
	print(best_negative)
'''
#-------------------------------------

#Improving the model

# Refining text for stop words(He,She. It, They, I, in, of, at) 
from nltk.corpus import stopwords
english_stop_words = stopwords.words('english')

def remove_stop_words(corpus):
	removed_stop_words = []
	for review in corpus:
		removed_stop_words.append(
			' '.join([word for word in review.strip().split()
			if word not in english_stop_words])		
		)
	return removed_stop_words

no_stop_words = remove_stop_words(reviews_train_clean)
#print("No stop words: {}\n".format(no_stop_words[0]))

X = cv.transform(no_stop_words)
final_model.fit(X,target)
print("Accuracy(C=0.05) with No Stop Words is: {}\n".format(accuracy_score(target,final_model.predict(X_test))))

print("\n\n")

#Normalization(convert all the diff. verb forms of a word into a single word)

#Two Methods are: Stemming and Lemmatization(much better and practical)

#stemming

def to_stem_reviews(corpus):
	from nltk.stem.porter import PorterStemmer
	stemmer = PorterStemmer()
	return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews = to_stem_reviews(reviews_train_clean)
#print("Stemmed Review: {}\n".format(stemmed_reviews[0]))

X = cv.transform(stemmed_reviews)
final_model.fit(X,target)
print("Accuracy(C=0.05) with Stemmed Reviews is: {}\n".format(accuracy_score(target,final_model.predict(X_test))))

print("\n\n")

def to_lemma_reviews(corpus):
	from nltk.stem import WordNetLemmatizer
	wnl = WordNetLemmatizer()
	return [' '.join([wnl.lemmatize(word) for word in review]) for review in corpus]

lemmatized_reviews = to_lemma_reviews(reviews_train_clean)
#print("Lemmatized Reviews: {}\n".format(lemmatized_reviews[0]))

X = cv.transform(lemmatized_reviews)
final_model.fit(X,target)
print("Accuracy(C=0.05) with Lemmatized Reviews is: {}\n".format(accuracy_score(target,final_model.predict(X_test))))

print("\n\n")

#ngrams can be used within Vectorizer defination with following defination
# ngram_cv = CountVectorizer(binary=True, ngrams=(1,2)|(2,2)) 
#ngram_cv.transform(reviews_train_clean)

#Instead of using single words as colums to form a sparse matrix, we can count words used in a reviews and word count(highest|lowest) can help us understand review better

wc_cv = CountVectorizer(binary=False)
wc_cv.fit(reviews_train_clean)
X = wc_cv.transform(reviews_train_clean)
X_test = wc_cv.transform(reviews_train_clean)

final_model.fit(X,target)
print("Accuracy(C=0.05) using Wordcount: {}".format(accuracy_score(target,final_model.predict(X_test))))

print("\n\n")

#Tf-Idf(Term frequency-inverse document frequency)
#it defines number of times a word appear in a document relative to number of documents it's present in

from sklearn.feature_extraction.text import TfidfVectorizer
tf_vec = TfidfVectorizer()
tf_vec.fit(reviews_train_clean)
X = tf_vec.transform(reviews_train_clean)
X_test = tf_vec.transform(reviews_test_clean)

print("Accuracy(C=0.05) using Tf-Idf: {}".format(accuracy_score(target,final_model.predict(X_test))))

#-------------------------------------------------------------------------------










