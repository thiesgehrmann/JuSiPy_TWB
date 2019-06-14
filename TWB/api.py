import TWB

from scipy.spatial.distance import cdist

class API(object):
    def __init__(self, tags, news_api_key):
        self.lsc = TWB.LSC()
        self.W = TWB.Wikipedia(tags)
        
        T = [ a.text() for a in self.W.articles ]
        
        self.N  = TWB.News(tags, news_api_key)
        self.TC = self.N.topic_countries
        
        self.D = TWB.Dictionary(T)
        self.A = self.D.annotate(T)
    #edef
    
    def get(self, xliffs, max_topics=10):
        """
        
        Get information about an xliff object/ list of xliff objects
        
        parameters:
        -----------
        xliffs: List[TWB.XLIFF] | XLIFF
        
        max_topics: Integer
            Maximum number of topics to report (sorted by prevalence)
        
        returns:
        dict
        """
        
        if isinstance(xliffs, TWB.XLIFF):
            xliffs = [ xliffs ]
        #fi
        
        XA = [ { 'target_lsc' : self.lsc.detect(i.target_lang) } for i in xliffs ]
        
        topic_languages = self._get_topics_languages(xliffs, max_topics)
        for i,v in enumerate(topic_languages):
            XA[i]['topics'] = v
        #efor
        
        #Your other annotations
        #o_annot = self._(xliffs)
        #for i,v in enumerate(o_annot):
        #   XA[i]['o_annot'] = v
        #efor
        
        return XA
    #edef
    
    def _get_topics_languages(self, xliffs, max_topics=10):
        """
        Get the topic and target language relevance for an xliff object
        A bit messy for now...
        
        parameters:
        -----------
        xliffs: List[TWB.XLIFF] | XLIFF
            List of xliffs
        
        max_topics: Integer
            Maximum number of topics to report (sorted by prevalence)
        
        returns:
        dict
        """
        
        doc_text  = [ ' '.join(x.source) for x in xliffs ]

        dists = cdist(self.D.annotate(doc_text), self.A)
        
        articles = self.W.articles

        xliff_topics = []

        for xliff_i, d_i in enumerate(dists):
            S = sorted(enumerate(d_i), key=lambda x:x[1])
            S = [ (articles[i].topic, d) for (i,d) in S ]
            T = { t: min(v) for (t,v) in TWB.common.group(S, key=lambda x: x[0], value=lambda x:x[1]).items()}
            T = dict(list(sorted(T.items(), key=lambda x:x[1]))[:max_topics])
            xliff_topics.append(T)
        #efor

        xliff_annot = []
        for xliff_i, xt, in enumerate(xliff_topics):
            A = { t : { 'distance' : d,
                        'relevant_country_languages' : {
                            country : { 'frequency' : count, 'language_relevant' : False }
                            for (country, count) in self.TC[t]
                        }
                      } for (t,d) in xt.items() }
                     
            target_lang = self.lsc.detect(xliffs[xliff_i].target_lang)
            for t in A:
                for country in A[t]['relevant_country_languages']:
                    A[t]['relevant_country_languages'][country]['language_relevant'] = target_lang.language.iso3 in self.lsc.country_languages(country)
                #efor
            #efor
            xliff_annot.append(A)
        #efor
        return xliff_annot
    #edef
#eclass