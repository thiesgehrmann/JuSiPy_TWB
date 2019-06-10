from lxml import etree as ET

"""
Some code to deal with the XLIFF formtted data.
See the spec at 
https://docs.oasis-open.org/xliff/v1.2/os/xliff-core.html
"""

def format_namespace_tag(tag, nsmap):
    """
    Because none of these fucking tools seem to do their job.
    Here a function to automatically determine the full tag name given a predefined namespace.

    parameters:
    -----------
    tag: String
        The name of the tag.
        A namespace can be specified with 'ns:tag'
        If no namespace is specified, then the default is used. If undefined, it is empty.
        If in the hierarchy, nsxml is specified, then it is the default.

        Other namespaces, e.g.
          tree <- <root nsxml:its='http://blahbla.com'></root>
        can be used like so:
          format_namespace_tag('its:a', tree.nsmap)
    nsmap: dict
        The dictionary of namespaces

    Returns:
        String    
    """
    s = tag.split(':')
    ns, t = (None, tag) if len(s) == 1 else s
    return '{%s}%s' % (nsmap.get(ns, ''), t)
#edef

class XLIFF(object):
    """
    Parse an XLIFF formatted file

    usage:

    x = XLIFF(file.xliff)

    x.filename # The name of the original file
    x.source   # The source text
    x.target   # The target text (if translated)
    x.source_lang # The source language
    x.target_lang # The target language

    """

    def __init__(self, filename):
        """
        Load an XLIFF formatted XML file

        """
        huge_parser = ET.XMLParser(encoding='utf-8', recover=True, huge_tree=True)
        self._xml  = ET.parse(filename , huge_parser) 
        self._root = self._xml.getroot()
        self._nsmap = self._root.nsmap
    #edef

    @property
    def _case_attributes(self):
        reltags = [ format_namespace_tag('internal-file', self._nsmap), 
                    format_namespace_tag('external-file', self._nsmap) ]
        for f in self._root:
            if len(list(f.iter(*reltags))) > 0:
                return f.attrib
            #fi
        #efor
        return {}
    #edef

    @property
    def filename(self):
        """
        Return the name of the original source file.
        It searches through each <file> tag for a child called internal-file or external-file.
        If this child tag exists, then it returns the file's 'original' attribute which contains the filename.

        returns:
            String
        """
        return self._case_attributes.get('original', None)
    #edef

    @property
    def source_lang(self):
        """
        Return the source language of this translation

        returns:
            String
        """
        return self._case_attributes.get('source-language', None)
    #edef

    @property
    def target_lang(self):
        """
        Return the target language of this translation
        
        returns:
            String
        """
        return self._case_attributes.get('target-language', None)
    #edef

    def _itertext(self, tag):
        text = [ "".join(s.itertext()).replace(u'\xa0', u' ')
                    for s in self._root.iter(format_namespace_tag(tag, self._nsmap)) ]
        return text
    #edef

    @property
    def source(self):
        """
        From an XLIFF formatted XML file, return the source text.

        returns:
            List[String]
        """

        return self._itertext('source')
    #edef

    @property
    def target(self):
        """
        From an XLIFF formatted XML file, return the source text.

        returns:
            List[String]
        """

        return self._itertext('target')
    #edef

#eclass

