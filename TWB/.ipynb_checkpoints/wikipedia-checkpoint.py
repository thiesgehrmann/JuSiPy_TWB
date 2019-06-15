import urllib
import hashlib
import requests
import json
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


# basically copied and modified from the News module. Bad form. No code reusability :S

def gen_request(query, **kwargs):
    """
    Generate a Wikipedia request, based on a given query
    
    parameters:
    -----------
    query: String
        The name of the wikipedia article
        
    **kwargs: Dict
        Additional arguments to the query
        
    returns:
        String. Request URL
    """
    url = 'https://en.wikipedia.org/w/api.php?'
    defaults = {'action' : 'query',
                'format' : 'json',
                'titles' : query,
                'prop'   : 'extracts'}
    defaults.update(kwargs)
    data = urllib.parse.urlencode(defaults) + "&exintro&explaintext"
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
    cache_file = '%s/../cache/wikipedia.%s.json' % (dir_path, hashlib.md5(request.encode('utf-8')).hexdigest())
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

def get_article(query, cache=True, overwrite=False, **kwargs):
    """
    Perform a Wikipedia query
    
    parameters:
    -----------
    query: String
        A proper query (see https://newsapi.org/docs/endpoints/everything)
    cache: Boolean
        Cache the results?
    overwrite: Boolean
        Overwrite the cache?
        
    **kwargs: Dict
        Additional arguments for gen_request
        
    returns:
        WikiArticle
    """
    
    req = gen_request(query, **kwargs)
    data = do_request(req, cache, overwrite)['query']['pages']
    data = data[list(data.keys())[0]]
    return WikiArticle(data, req)
#edef

class WikiArticle(object):
    """
    Handle a Wikipedia article.
    
    available properties:
        * content
        * title
        
    available methods:
        * text
    """
    def __init__(self, article, request):
        self._request = urllib.parse.parse_qs(request.split('?')[1])
        self._article = article
        
        if 'extract' not in article:
            raise ValueError('"extract" is not in the article "%s". Check return value.' % self.title)
        #fi
    #edef
    
    @property
    def content(self):
        """
        Return the content of the article
        """
        return self._article['extract']
    #edef
    
    @property
    def title(self):
        """
        Return the title of the article
        """
        return self._article['title'].lower()
    #edef

    
    def text(self, content=True, title=True):
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
        
        return ' '.join(T).upper()
    #edef
    
    @property
    def topic(self):
        """
        Return the topic by which this article was found.
        """
        return self.title
    #edef
    
    def __equals__(self, other):
        return self._article.text() == other._article.text()
    #edef
#eclass

class Wikipedia(object):
    """
    An object to handle news articles and data about them
    """
    
    def __init__(self, topics, **kwargs):
        """
        Initialize the News articles
        
        parameters:
        -----------
        
        topics: list[String]
            The news topics you are interested in
            
        **kwargs: Dict
            Additional arguments to news_articles
        """
        self._kwargs  = kwargs
        self._topic_articles = {}
        for topic in topics:
            self.add_topic(topic.lower())
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
            Additional arguments to get_article
        """
        
        kw = self._kwargs
        kw.update(kwargs)
        self._topic_articles[topic] = [ get_article(topic, **kw) ]
        
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
