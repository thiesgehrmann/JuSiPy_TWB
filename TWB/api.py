import TWB

from scipy.spatial.distance import cdist

class API(object):
    def __init__(self, tags, news_api_key):
        self.lsc = TWB.LSC()
        self.W = TWB.Wikipedia(tags)
        
        T = [ a.text() for a in self.W.articles ]
        

        self.trends = { t: TWB.Trends(None, topic=t) for t in tags }
        
        self.N  = TWB.News(tags, news_api_key)
        self.TC = self.N.topic_countries
        
        self.D = TWB.Dictionary(T)
        self.A = self.D.annotate(T)
    #edef
    
    def get(self, xliff, timestamp, max_topics=10):
        """
        
        Get information about an xliff object
        
        parameters:
        -----------
        xliff: XLIFF
        
        max_topics: Integer
            Maximum number of topics to report (sorted by prevalence)
        
        returns:
        dict
        """
        
        if not isinstance(xliff, TWB.XLIFF):
            raise valueError('Expecting a single XLIFF.')
        #fi
        
        XA =  { 'target_lsc' : self.lsc.detect(xliff.target_lang),
                 'source_lsc' : self.lsc.detect(xliff.source_lang) } 
        
        XA['topics'] = self._get_topics_languages(xliff, timestamp, max_topics)
        
        #Your other annotations
        #XA['o_annot'] = self._other_annot(xliffs)

        
        return XA
    #edef
    
    def _get_topics_languages(self, xliff, timestamp, max_topics=10):
        """
        Get the topic and target language relevance for an xliff object
        A bit messy for now...
        
        parameters:
        -----------
        xliffs: XLIFF
            List of xliffs
        
        max_topics: Integer
            Maximum number of topics to report (sorted by prevalence)
        
        returns:
        dict
        """
        
        xliffs = [ xliff ]
        
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
            target_lang = self.lsc.detect(xliffs[xliff_i].target_lang)
            source_lang = self.lsc.detect(xliffs[xliff_i].source_lang)
            A = { t : { 'distance' : d,
                        'news_country_languages' : {
                            country : {
                              'frequency' : count,
                              'target_rel' : target_lang.language.iso3 in self.lsc.country_languages(country),
                              'source_rel' : source_lang.language.iso3 in self.lsc.country_languages(country)
                            }
                            for (country, count) in self.TC[t]
                            if (target_lang.language.iso3 in self.lsc.country_languages(country)) # or (source_lang.language.iso3 in self.lsc.country_languages(country))
                        },
                        'trends_country_languages' : {
                            country : {
                              'frequency' : count,
                              'target_rel' : target_lang.language.iso3 in self.lsc.country_languages(country),
                              'source_rel' : source_lang.language.iso3 in self.lsc.country_languages(country)
                            }
                            for (country, count) in list(dict(self.trends[t].ranked_countries_per_topic(timestamp)[:10]).items())
                            if (target_lang.language.iso3 in self.lsc.country_languages(country)) # or (source_lang.language.iso3 in self.lsc.country_languages(country))
                        }
                      }
                 for (t,d) in xt.items()
                }
            
            xliff_annot.append(A)
        #efor
        return xliff_annot[0]
    #edef
#eclass