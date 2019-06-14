import requests
import urllib
import json
import os
import hashlib
import TWB
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

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
    def __init__(self, article, request, lsc=TWB.LSC()):
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
    
    def __equals__(self, other):
        return self._article == other._article
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
                'q'        : query,
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
    print(cache_file)
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
