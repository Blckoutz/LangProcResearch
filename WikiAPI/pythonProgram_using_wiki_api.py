import requests
from lxml import html
url = 'https://en.wikipedia.org/w/api.php'

##Clarify the one we need
subject=input('What subject do you want information on?')
#1. check the wiki name
#make a search
#find occurrences and check if an entity exists and what is the subject name
params = {
            'action':'query',
            'format':'json',
            'list':'search',
            'utf8':1,
            'srsearch':subject
        }
 
data = requests.get(url, params=params).json()
#print(len(data['query']['search']))
subjectConfirmed=0
count=0
if len(data['query']['search'])> 1:
    for i in data['query']['search']:
        print('CHOICE',count,i['title'], ' - Word count: ', i['wordcount'])
        count=count+1
    subjectConfirmed=input('Please enter the number of the topic you were requesting')
    subject=data['query']['search'][int(subjectConfirmed)]['title']
#print(subjectConfirmed)    
#print(data['query']['search'][int(subjectConfirmed)]['title'])

#2. grab the document
params = {
        'action': 'parse',
        'format': 'json',
        'page': subject,
        'prop': 'text',
        'redirects':''
    }
  
response = requests.get(url, params=params).json()
raw_html = response['parse']['text']['*']
#print(raw_html)
document = html.document_fromstring(raw_html)
#print(document) 
text = ''
for p in document.xpath('//p'):
   text += p.text_content() + '\n'
#print(len(text))
print(text[:])

#3. now we can store this data, process it, etc.
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords #stopword handling
from nltk import pos_tag,wordpunct_tokenize

english_stops = set(stopwords.words('english'))

words=word_tokenize(text) 
sentences=sent_tokenize(text)
#TODO: Decide on a tokenizer
print('# words',len(words))#has commas, ( etc.
print('# sentences',len(sentences))
withoutStops=[word for word in words if word not in english_stops]
print('# words without stop words',len(withoutStops))

# def tok(self,fileids=None,categories=None):
 #   for para in self.paras(fileids=fileids):
  #      yield [
    #        pos_tag(wordpunct_tokenize(sent))
   #         for sent in sent_tokenize(para)
    #    ]

#tok(text)

for sent in sent_tokenize(text):
    pos_tag(wordpunct_tokenize(sent))

