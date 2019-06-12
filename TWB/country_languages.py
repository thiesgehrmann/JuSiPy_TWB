import pandas as pd
from .lsc import LSC
from .common import *
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class CountryLanguages(object):
    """
    An index of languages spoken in each country
    """
    def __init__(self):
        lsc = LSC()
        cl = df_to_upper(pd.read_excel('%s/../data/country_languages.xlsx' % dir_path))
        cl['country_iso3'] = cl.country_iso2.apply(lambda x: lsc.country(x).iso3)
        cl['languages_iso3'] = cl.languages_iso.apply(lambda x: [
           lsc.language(l.strip()).iso3 for l in x.split(',')
        ])
        self.CL = cl[['country_iso3', 'languages_iso3']]
    #edef
    
    def country(self, country):
        """
        Lookup the languages spoken in a country from the iso3 code
        
        parameters:
        -----------
        country: String
            ISO3 formatted country code
            
        returns:
        --------
        List[String]
            list of languages in ISO3 format
        """
        country = country.upper()
        r = self.CL[self.CL.country_iso3 == country].languages_iso3.values
        if len(r) == 0:
            return []
        #fi
        return r[0]
    #edef
    
    def is_language_relevant(self, country, language):
        """
        Check if a language is spoken in a specific country.
        
        parameters:
        -----------
        country: String.
            ISO3 country
            
        language: String
            ISO3 language ID
        
        returns:
        --------
        Boolean"""
        lang = self.country(country)
        return language.upper() in lang
    #edef
        