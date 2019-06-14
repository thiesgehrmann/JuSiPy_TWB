import requests
import urllib
import json
import os
import hashlib
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

from .lsc import LSC
from . import common

class Article(object):
    """
    Handle a news article.
    
    available properties:
        * content
        * title
        * description
        
    available methods:
        * text
        * country
    """
    def __init__(self, article, request, lsc=LSC()):
        self._request = urllib.parse.parse_qs(request.split('?')[1])
        self._article = article
        self._countries = None
        self._lsc = lsc
    #edef
    
    @property
    def content(self):
        """
        Return the content of the article
        """
        return self._article['content']
    #edef
    
    @property
    def title(self):
        """
        Return the title of the article
        """
        return self._article['title']
    #edef
    
    @property
    def description(self):
        """
        Return the title of the article
        """
        return self._article['description']
    #edef
    
    def text(self, content=True, description=True, title=True):
        """
        Get text from the article as a single string
        
        parameters:
        -----------
        content: Boolean
            Return the content of the article
        
        description: Boolean
            Return the description of the article
            
        title: Boolean
            Return the title of the article
            
        returns:
        --------
        text: String
            String of the article text
        """
        T = []
        if content and (self.content is not None):
            T.append(self.content)
        #fi
        
        if title and (self.title is not None):
            T.append(self.title)
        #fi
        
        if description and (self.description is not None):
            T.append(self.description)
        #fi
        
        return ' '.join(T).upper()
    #edef
        
        
    def country(self, content=True, description=True, title=True):
        """
        Identify the country relevant in an article
        
        parameters:
        -----------
        content: Boolean
            Look in the content of the article
        
        description: Boolean
            Look in the description of the article
            
        title: Boolean
            Look in the title of the article
            
        returns:
        --------
        List[String] : ISO3 formats of countries mentioned in the 
        """
        
        if self._countries is not None:
            return self._countries
        #fi
        
        text = self.text(content, description, title)
        
        self._countries = []
        for c in list(self._lsc.CC.country.values): # Pretty intensive search... think of a better way...
            if c in text:
                self._countries.append(self._lsc.country(c, format='country').iso3)
            #fi
        #efor
        return self._countries
    #edef
    
    @property
    def topic(self):
        """
        Return the topic by which this article was found.
        """
        return self._request['q'][0].lower()
    #edef
    
    def __equals__(self, other):
        return self._article.text() == other._article.text()
    #edef
#eclass

def gen_request(query, api_key, **kwargs):
    """
    Generate a newsAPI request, based on a given query
    
    parameters:
    -----------
    query: String
        A proper query (see https://newsapi.org/docs/endpoints/everything)
    api_key: String
        A newsAPI API Key
        
    returns:
        String. Request URL
    """
    url = 'https://newsapi.org/v2/everything?'
    defaults = {'pageSize' : 100,
                'language' : 'en',
                'sortBy'   : 'relevancy',
                'q'        : query.lower(),
                'apiKey'   : api_key}
    defaults.update(kwargs)
    data = urllib.parse.urlencode(defaults)
    return url + data
#edef

def do_request(request, cache=True, overwrite=False):
    """
    Perform a newsAPI query
    
    parameters:
    -----------
    query: String
        A proper request from gen_request

    cache: Boolean
        Cache the results?
    overwrite: Boolean
        Overwrite the cache?
        
    returns:
        Dict[Dict] of JSON result
    """
    cache_file = '%s/../cache/newsapi.%s.json' % (dir_path, hashlib.md5(request.encode('utf-8')).hexdigest())
    if cache and os.path.isfile(cache_file) and (not overwrite):
        with open(cache_file) as ifd:
            return json.load(ifd)
        #ewith
    #fi

    response = requests.get(request)
    if response.status_code != 200:
        raise RuntimeError('Unsuccessful[%d]: %s' % (response.status_code, request))
    #fi
    
    data = response.json()
    
    if cache or overwrite:
        with open(cache_file, 'w') as ofd:
            json.dump(data, ofd)
        #ewith
    #fi
    
    return data
#edef
        

def news_articles(query, api_key, pages=1, cache=True, overwrite=False, **kwargs):
    """
    Perform a newsAPI query, for a specified number of pages
    
    parameters:
    -----------
    query: String
        A proper query (see https://newsapi.org/docs/endpoints/everything)
    api_key: String
        A newsAPI API Key
    pages: Integer
        The total number of pages to get
    cache: Boolean
        Cache the results?
    overwrite: Boolean
        Overwrite the cache?
        
    returns:
        List[Article] of articles
    """
    A = []
    for page in range(pages):
        req  = gen_request(query, api_key, page=page+1, **kwargs)
        data = do_request(req, cache, overwrite)
        A.extend([Article(a, req) for a in data['articles']])
    #efor
    return A
#edef


        
class News(object):
    """
    An object to handle news articles and data about them
    """
    
    def __init__(self, topics, api_key, **kwargs):
        """
        Initialize the News articles
        
        parameters:
        -----------
        
        topics: list[String]
            The news topics you are interested in
            
        api_key: String
            The newsAPI api key
            
        **kwargs: Dict
            Additional arguments to news_articles
        """
        self._topics  = topics
        self._api_key = api_key
        self._kwargs  = kwargs
        self._topic_articles = {}
        for topic in topics:
            self.add_topic(topic)
        #efor
        
    #edef
    
    def add_topic(self, topic, **kwargs):
        """
        Add a topic of interest 
        
        parameters:
        -----------
        
        topic: String
            The news topic you are interested in
            
        **kwargs: Dict
            Additional arguments to news_articles
        """
        
        kw = self._kwargs
        kw.update(kwargs)
        self._topic_articles[topic] = news_articles(topic, self._api_key, **kw)
    #edef
    
    @property
    def topic_countries(self):
        """
        Get a list of countries per topic, sorted by prevalence
        
        returns:
        --------
        dict[topic->List[(ISO3, frequency)]]
        """
        TC = { t: common.freq([i for l in [a.country() for a in A] for i in l])
                  for (t,A) in self._topic_articles.items() }
        return { t: sorted(v.items(), key=lambda x: x[1], reverse=True) for (t,v) in TC.items() }
    #edef
    
    @property
    def articles(self):
        """
        Return a list of articles
        """
        return [ a for topic in self._topic_articles for a in self._topic_articles[topic] ]
    #edef
    
    @property
    def topics(self):
        """
        Get the topics
        
        returns:
        --------
        Dict[topic->List[Article] (sorted by most relevant to topic)]
        """
        return self._topic_articles
    #edef
#edef
