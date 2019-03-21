from bs4 import BeautifulSoup
import random


f = open("../Downloads/test/newstest2018-ende-src.en.sgm",'r')
html_doc = f.read()
soup = BeautifulSoup(html_doc, 'html.parser')
a = soup.find_all(id)
out = [x for x in a[0].text.split('\n') if x != ''][:]

# random.Random(4).shuffle(out)

english_test_sentences = out

f = open("../Downloads/test/newstest2018-ende-ref.de.sgm",'r')
html_doc = f.read()
soup = BeautifulSoup(html_doc, 'html.parser')
a = soup.find_all(id)
out = [x for x in a[0].text.split('\n') if x != ''][:]
# random.Random(4).shuffle(out)
german_test_sentences = out

# pairs = list(zip(english_test_sentences,german_test_sentences))
# random.Random(4).shuffle(pairs)
# english_test_sentences,german_test_sentences = zip(*pairs)


# print(english_test_sentences[:10])
# print(german_test_sentences[:10])