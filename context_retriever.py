import requests  # Getting text from websites
import html2text  # Converting wiki pages to plain text
from googlesearch import search  # Performing Google search
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from markdown import markdown
import wikipedia


# DO GOOGLE SEARCH AND RETURN n RELEVANT URLS
def query_pages(query, n):
    urls = search(query, num_results=n)
    return urls


# FUNCTIONS TO GET TEXT FROM RELEVANT WEB PAGES INTO A LIST OF CONTEXTS
# Convert a markdown string into a plaintext
def markdown_to_text(markdown_string):
    html = markdown(markdown_string)

    # Remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', '', html)
    html = re.sub(r'<code>(.*?)</code>', '', html)

    # Extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text


def format_text(text):
    text = markdown_to_text(text)
    text = text.replace('\n', ' ')
    return text


# Return a list of converted contexts
def query_to_text(query, n):
    html_conv = html2text.HTML2Text()
    html_conv.ignore_links = True
    html_conv.escape_all = True

    # Get the entire HTML from the webpage
    text = []
    for link in query_pages(query, n):  # [0:n]
        if 'https://' in link:
            try:
                req = requests.get(link)
                if req.status_code == 200:
                    text.append(html_conv.handle(req.text))
                    text[-1] = format_text(text[-1])
                    print(link)
            except:
                pass

    return text


# DO WIKIPEDIA SEARCH AND RETURN A SUMMARY OF A RELEVANT PAGE
def search_wiki (query):
    try:
        summary = wikipedia.summary(query)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return "Ambiguous"


def summarize_context(context):
    # Tokenizing the context
    the_stopwords = set(stopwords.words("english"))
    words = word_tokenize(context)

    # Creating a frequency table to keep the score of each words.
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in the_stopwords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score of each sentence
    sentences = sent_tokenize(context)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text
    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence

    return summary
