from collections import namedtuple
from .common import *
import numpy as np
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class CountryCode(object):
    """
    Translate between country codes
    Usage:
        CC = CountryCode()
        CC['Germany'] # Returns: countrycode(country='Germany', iso2='DE', iso3='DEU', number=276)
    """

    __slots__ = [ '_matrix', '_index', '_tups', '_special_cases' ]

    def __init__(self):
        self._matrix = pd.read_excel('%s/../data/country_codes.xlsx' % dir_path)
        tup = namedtuple('countrycode', self._matrix.columns)
        self._tups = list(map(lambda x: tup(*x), self._matrix.values))
        self._index = { str(k).lower() : i for i,t in enumerate(self._tups) for k in t }

        self._special_cases = { 'democratic republic of the congo': 'congo, (kinshasa)',
                                "democratic people's republic of korea": 'korea (south)',
                                'republic of korea': 'korea (south)',
                                'the former yugoslav republic of macedonia': 'macedonia, republic of',
                                'east timor': 'timor-leste',
                                'burma': 'myanmar',
                                'dutch': 'netherlands',
                                'tanzania': 'tanzania, united republic of',
                                'russia': 'russian federation',
                                'uae': 'united arab emirates',
                                'south korea': 'korea (south)',
                                'north korea': 'korea (north)',
                                'taiwan': 'taiwan, republic of china',
                                'venezuela': 'venezuela (bolivarian republic)'}

        for s,c in self._special_cases.items():
            self._index[s.lower()] = self._index[c.lower()]
        #efor
        self._tups.append(tup(None, None, None, None))
    #edef

    @property
    def country(self):
        """
        A list of all countries represented in this index
        """
        return [ cc.country for cc in self._tups]
    #edef

    @property
    def iso2(self):
        """
        A list of all ISO2 represented in this index
        """
        return [ cc.iso2 for cc in self._tups]
    #edef

    @property
    def iso3(self):
        return [ cc.iso2 for cc in self._tups]
    #edef

    @property
    def number(self):
        return [ cc.number for cc in self._tups]
    #edef

    def _lookup(self, key):
        k = str(key).lower()

        if k in self._index:
            return self._tups[self._index[k]]
        #fi

        matches = difflib.get_close_matches(k, self._index.keys())
        if len(matches) > 0:
            km = matches[0]
            return self._tups[self._index[km]]
        else:
            sk = ' '.join(k.split()[:-1])
            if len(sk) == 0:
                return self._tups[-1]
            #fi
            rec = self._lookup(sk)
            if rec.country is None:
                print('Did not find "%s". Consider adding special case' % key)
            #fi
            return rec
        #fi
    #edef

    def __getitem__(self, keys):
        if isinstance(keys, str) or isinstance(keys, int):
            return self._lookup(keys)
        else:
            return [ self._lookup(k) for k in keys ]
        #fi
    #edef
#edef

class LSC(object):
    """
    An object to handle the lookup and conversion of language-script-country codes
    
    lsc = LSC()
    
    lsc.detect('sr-Latn-RS')
    lsc.country()
    """
    def __init__(self):
        self.LC = df_to_upper(pd.read_excel('%s/../data/language_codes.xlsx' % dir_path))
        self.CL = df_to_upper(pd.read_excel('%s/../data/country_languages.xlsx' % dir_path))

        self.CC = df_to_upper(pd.read_excel('%s/../data/country_codes.xlsx' % dir_path))
        self._CC = CountryCode()
        self.SC = df_to_upper(pd.read_excel('%s/../data/script_codes.xlsx' % dir_path))
    #edef
    
    def country(self, code, format=None):
        """
        Lookup country by code
        
        parameters:
        -----------
        code: string | int
            The code to lookup
        format: None | String in [iso2, iso3, un, country]
            The format to use. If None, it is detected
            
        returns:
        --------
        namedtuple(country, iso2, iso3, un)
        """
        if format is None:
            if isinstance(code, int):
                format = 'un'
            elif isinstance(code, str) and code.isdigit():
                format = 'un'
            elif isinstance(code, str) and (len(code) == 2):
                code = code.upper()
                format = 'iso2'
            elif isinstance(code, str) and (len(code) == 3):
                code = code.upper()
                format = 'iso3'
            else:
                code = code.upper() if isinstance(code, str) else None
                format = 'country'
            #fi
        #fi
        
        if code is None:
            return namedtuple('country', self.CC.columns)(*([None] * len(self.CC.columns)))
        
        if format == 'country':
            return self._CC[code]
        #fi
        
        r = self.CC[self.CC[format] == code].values
        if len(r) == 0:
            r = [[None] * len(self.CC.columns)]
        #fi
        return namedtuple('country', self.CC.columns)(*r[0])
    #edef
        
    def script(self, code, format=None):
        """
        Lookup script by code
        
        parameters:
        -----------
        code: string | int
            The code to lookup
        format: None | String in [code, number]
            The format to use. If None, it is detected
            
        returns:
        --------
        namedtuple(code, number, name, alias, direction, version, characters, remarks)
        """
        if format is None:
            if isinstance(code, int):
                format = 'number'
            elif isinstance(code, str) and code.isdigit():
                format = 'number'
            elif isinstance(code, str):
                code = code.upper()
                format = 'code'
            #fi
        #fi
        
        r = self.SC[self.SC[format] == code].values
        if len(r) == 0:
            r = [[None] * len(self.SC.columns)]
        #fi
        return namedtuple('script', self.SC.columns)(*r[0])
    #edef

        
    def language(self, code, format=None):
        """
        Lookup script by code
        
        parameters:
        -----------
        code: string | int
            The code to lookup
        format: None | String in [iso2, iso3, english, french, german]
            The format to use. If None, it is detected
            
        returns:
        --------
        namedtuple(iso3, iso2, english, french, german)
        """
        code = str(code).upper()
        if format is None:
            if len(code) == 2:
                code = code.upper()
                format = 'iso2'
            elif len(code) == 3:
                code = code.upper()
                format = 'iso3'
            else:
                code = code.upper()
                format = 'iso3'
            #fi
        #fi
        
        r = self.LC[self.LC[format] == code].values
        if len(r) == 0:
            r = [[None] * len(self.LC.columns)]
        #fi
        return namedtuple('language', self.LC.columns)(*r[0])
    #edef
    
    def detect(self, code):
        """
        Breakdown a language[-script]-(locale/country) encoded string into a the ISO specification
        
        parameters:
        -----------
        code: String
            The code. It will fit into one of the three formats:
            * language                : script -> LATN, country->None
            * language-country        : script -> LATN
            * language-script-country
        
        returns:
        --------
        namedtuple(language, script, country)
            language, script and country are again namedtuples with information from the
            `language`, `script`, and `country` methods.
        """
        fields = str(code).lower().split('-')
        lang = None
        script = None
        country = None

        if len(fields) == 1:
            lang, = fields
            script = 'Latn'
        elif len(fields) == 2:
            lang, country = fields
            script = 'Latn'
        elif len(fields) == 3:
            lang, script, country = fields
        else:
            raise ValueError("Invalid code")
        #fi
        
        return namedtuple('code', ['language', 'script', 'country'])(self.language(lang), self.script(script), self.country(country))
    #edef
#eclass