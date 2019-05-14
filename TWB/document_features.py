import scipy
import spacy
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralCoclustering

#########################################################################

def is_token_interesting(token):
    """
    How do we define a token as interesting?
    In this function, We define it as interesting if it satisfies the following conditions
    * It is not a stopword
    * It is not identical to its head word in the grammatical structure
    * Both it, AND the head word are verbs or nouns.
        * i.e. we are not interested in adverbs, but we are specifically interested in verbs that act on nouns, or vice versa.

    parameters:
    -----------
    token: spacy.tokens.token.Token
        The token to evaluate

    Output:
        Boolean
    """
    if len(token) < 2:
        return False
    if token.is_stop:
        return False
    elif token == token.head:
        return False
    elif (token.pos_ in [ 'NOUN', 'VERB']):
        return True
    #fi
    return False
#edef

#########################################################################

def gen_tokensets(documents, func=is_token_interesting, max_sents=None, nlp=None):
    """
    Generate the tokensets for a given document.

    parameters:
    -----------
    document: List[String] | String.
        The document(s) in a string format
        
    func: Function(spacy.tokens.token.Token) -> Boolean
        Decide whether to keep the token or not
        
    max_sents: None | Integer
        The maximum number of sentences to use. None implies no limit.
        
    nlp: None | String | spacy.lang.*
        The Spacy language model
        None -> Use the english language model
        String -> Load a specific language model
        spacy.lang.* -> Use a pre-loaded language model
        
        
    Returns:
        List[List[set[String]]]
    """
    
    nlp = spacy.load('en_core_web_sm') if nlp is None else spacy.load(nlp) if isinstance(nlp, str) else nlp
    
    def gen_tokensets_single(document, func, max_sents):
        doc = nlp(document)
        S = []
        for i, s in enumerate(doc.sents):
            S.append(set([t.lemma_ for t in s if func(t) ]))
            if i > max_sents:
                break
            #fi
        #efor
        return S
    #edef

    if max_sents is None:
        max_sents = np.inf
    #fi

    if isinstance(documents, str):
        documents = [ documents ]
    #fi

    T = []
    for i, d in enumerate(documents):
        print("\r%d/%s" % (i+1, str(len(documents)) if hasattr(documents, '__len__') else '?'), end='')
        T.append(gen_tokensets_single(d, func, max_sents))
    #efor
    print()

    return T
#edef

#########################################################################

def network_from_tokensets(tokensets, min_docs=4, min_tokens=1):
    """
    Given a list of sets of interesting tokens, generate a pandas dataframe of setsXtokens, indicating presents/absense
    
    parameters:
    -----------
    sets: List[List[Set[str]]]
        For each document, a list of sets (each which represents a sentence) of tokens.

    min_docs: Integer
        The minimum number of documents required to retain a token in the network (default 4)

    min_tokens: Integer
        The minimum number of tokens required to retain a sentence in the network (default 1)

    Returns:
        pd.DataFrame with 2+n_tokens columns and n_sentences rows.
        The first two columns, `document_` and `sentence_` represent the input indexes of the corresponding
        document and sentences.
    """
    
    uniq  = list(set.union(*[ set(s) for d in tokensets for s in d ]))
    uniqa = np.array(uniq)
    
    # A mapping from tokens to integers Offset of 2 for doc and sentence ID
    M = { t:i+2 for i,t in enumerate(uniq) }
    
    # Generate a matrix of zero counts
    # The first two columns are for the document and sentence IDs
    P = np.zeros((sum([len(s) for s in tokensets]), 2+len(uniq)))
    
    # Fill the first two columns with document and sentence IDs
    P[:,[0,1]] = [ [i,j] for i in range(len(tokensets)) for j in range(len(tokensets[i])) ]
    
    # Fill the matrix, looping over the sets
    k = 0
    for i, doc in enumerate(tokensets):
        for j, s in enumerate(doc):
            tok_ids = [ M[t] for t in s ]
            P[k,tok_ids] = 1
            k = k+1
        #efor
    #efor
    
    meta = ['document_', 'sentence_']
    N = pd.DataFrame(P, columns=meta + uniq)
    rel_uniq = list(N.columns[2:][(N.drop(columns='sentence_').groupby('document_').agg(sum) > 0).sum() > min_docs])
    N = N[meta + rel_uniq]
    N = N.loc[N[rel_uniq].sum(axis=1) >= min_tokens]
    return N
#edef

#########################################################################

def is_network(N):
    """
    Check if a pandas dataframe is a valid network.
    Checks if `document_` and `sentence_` are in column positions 0 and 1, respectively.
    Raises error if not.
    
    parameters:
    -----------
    N: pandas.DataFrame
        A pandas dataframe
        
    Returns:
    --------
    network: Boolean
        Is the dataframe a network?
    """
    return (N.columns[0] == 'document_') and (N.columns[1] == 'sentence_')
#edef

#########################################################################

def is_annotation(A):
    """
    Check if a pandas dataframe is a valid annotation.
    Checks if `document_` is in column position 0.
    Raises error if not.
    
    parameters:
    -----------
    A: pandas.DataFrame
        A pandas dataframe
        
    Returns:
    --------
    network: Boolean
        Is the dataframe an annotation?
    """
    return (A.columns[0] == 'document_')
#edef

#########################################################################

def _meta(NA):
    """
    Returns the metadata of a network or an annotation
    Assumes the network/annotation is already validated

    parameters:
    -----------
    NA: pandas.DataFrame
        A Network or an Annotation
        
    Returns:
    --------
    pandas dataframe with only the metadata
    """
    idx_offset = 1 + (NA.columns[1] == 'sentence_')
    return NA[NA.columns[:idx_offset]]
#edef

#########################################################################

def _features(NA):
    """
    Returns the features of a network or an annotation
    Assumes the network/annotation is already validated

    parameters:
    -----------
    NA: pandas.DataFrame
        A Network or an Annotation
        
    Returns:
    --------
    pandas dataframe with only the features
    """
    idx_offset = 1 + (NA.columns[1] == 'sentence_')
    return NA[NA.columns[idx_offset:]]
#edef

#########################################################################

def validate(NA, network=None):
    """
    Validate a dataframe as either a Network or an Annotation
    Raises error if not the case.
    
    parameters:
    -----------
    NA: pd.DataFrame
        The network or annotation dataframe to test
    network: None | True | False
        None -> Test if NA is a network OR an annotation
        True -> Test if NA is a network
        False -> Test if NA is an annotation
        
    Return:
    -------
    network: Boolean
        True if NA is a network
        False if NA is an annotation
        
    Raises
    """
    
    is_n = is_network(NA)
    is_a = is_annotation(NA)
    
    
    if network is None:
        if not (is_n or is_a):
            raise ValueError("The object is not a network or an anootation.")
        else:
            return is_n
        #fi
    #fi
    
    if network:
        if not is_n:
            raise ValueError("The object is not a network.")
        #fi
        return True
    else:
        if not is_a:
            raise ValueError("The object is not an annotation.")
        #fi
        return False
    #fi
#edef

#########################################################################
        
def is_collapsed(A):
    """
    Is an annotation collapsed?
    
    parameters:
    -----------
    A: pandas.DataFrame
        A pandas dataframe
        
    Returns:
    --------
    collapsed: Boolean
        Is the dataframe collapsed? (no 'sentence_' column in position 1)
    """
    
    validate(A, False)
    
    return not (('document_' == A.columns[0]) and ('sentence_' == 'A.columns[1]'))
#edef

#########################################################################

def feature_type(NA):
    """
    Return the type of the features encoded in the dataframe

    parameters:
    -----------
    NA: pandas.DataFrame
        A Network or an Annotation
        
    Returns:
    --------
    tokens: Boolean
        Are the features in this network tokens?
        True if it is the case.
        False if they are concepts
    """
    
    validate(NA)
        
    return not ('_' in ''.join(NA.columns[1 + (1 if ('sentence_' == NA.columns[1]) else 0):]))
#edef

#########################################################################

def collapse(A):
    """
    Collapse all sentences into a single document.
    The counts of features (tokens/concepts) are summed up across sentences within a document

    parameters:
    -----------
    A: pandas.DataFrame
        Dataframe of feature annotations, with at least columns `document_` and `_sentences`.

    Returns:
    C: pandas.DataFrame
        The annotations with counts across sentences within a sentence summed up.
        The `sentences_` column is removed.
    """

    if is_collapsed(A):
        return A.copy()
    #fi

    return self._obj.drop(columns='sentence_').groupby('document_').agg(sum).reset_index()
#edef

#########################################################################

def extract_concepts(N, method, *pargs, **kwargs):
    """
    Extract concepts from the network.
    
    parameters:
    -----------
    N: pd.DataFrame
        The network to extract concepts from

    *pargs, **kwargs:
        Additional arguments for SpectralCoclustering
    
    Returns:
    --------
    List[Set[String]]
        Each set represents a concept
    """
    
    validate(N, True)

    CC = SpectralCoclustering(*pargs, **kwargs).fit(N)
    C  = pd.DataFrame(np.concatenate([_meta(N).values, CC.rows_.transpose()], axis=1),
                      columns=list(_meta(N).columns) + [ '_'.join(N.columns[c]) for c in CC.columns_ ])
    
    return C
#edef

#########################################################################

def idf(N):
    """
    Calculate the IDF of tokens in a given corpus
    parameters
    """
    
    validate(N)
    
    X = _features(collapse(N))
    return np.log(X.shape[0] / (1+(X>1).sum()))
#edef

#########################################################################

def tfidf(N):
    """
    Calculate the TFIDF for a given corpus.
    """
    X = _features(N.collapse())
    idf = np.log(X.shape[0] / (1+(X>1).sum()))
    return X.divide(X.sum(axis=1), axis=0) * idf
#edef

#########################################################################

def annotate_tokens(N, tokensets):
    """
    Annotate tokensets given a set of tokens that are defined to be interesting

    parameters:
    -----------
    N: pd.DataFrame
        The network with relevant tokens
    tokensets: List[List[Set[String]]]
        Tokensets generated for a set of documents with gen_tokensets

    Returns:
        DataFrame that can be used with the `x.twb_accessors`
    """
    uniqs = set(_features(N).columns)
    uniq  = list(uniqs)
    uniqa = np.array(uniq)

    # A mapping from tokens to integers Offset of 2 for doc and sentence ID
    M = { t:i+2 for i,t in enumerate(uniq) }

    # Generate a matrix of zero counts
    # The first two columns are for the document and sentence IDs
    P = np.zeros((sum([len(s) for s in tokensets]), 2+len(uniq)))

    # Fill the first two columns with document and sentence IDs
    P[:,[0,1]] = [ [i,j] for i in range(len(tokensets)) for j in range(len(tokensets[i])) ]

    # Fill the matrix, looping over the sets
    k = 0
    for i, doc in enumerate(tokensets):
        for j, s in enumerate(doc):
            tok_ids = [ M[t] for t in (set(s) & uniqs)  ]
            P[k,tok_ids] = 1
            k = k+1
        #efor
    #efor

    meta = list(_meta(N).columns)

    F = pd.DataFrame(P, columns=meta + uniq)
    return F
#edef

#########################################################################

def annotate_concepts(N, tokensets):
    """
    Annotate tokensets given a set of concepts that are defined to be interesting

    parameters:
    -----------
    N: pd.DataFrame
        The network with relevant concepts
    tokensets: List[List[Set[String]]]
        Tokensets, produced by gen_tokensets
        
    Returns:
    --------
    A: pd.DataFrame
        An annotation of concepts per sentence
    """
    concepts = [ set(x.split('_')) for x in _features(N).columns ]
    uniq     = set.union(*concepts)

    P = np.zeros((sum([len(s) for s in tokensets]), 2+len(concepts)))

    # Fill the first two columns with document and sentence IDs
    P[:,[0,1]] = [ [i,j] for i in range(len(tokensets)) for j in range(len(tokensets[i])) ]

    # Fill the matrix, looping over the sets
    k = 0
    for i, doc in enumerate(tokensets):
        for j, s in enumerate(doc):
            s = set(s) & uniq
            concept_ids = [ c.issubset(s) for c in concepts ]
            P[k,concept_ids] = 1
            k = k+1
        #efor
    #efor

    meta = list(_meta(N).columns)
    F = pd.DataFrame(P, columns=meta + ['_'.join(sorted(list(c))) for c in concepts])
    return F
#edef

#########################################################################

def pdist(A, metric='euclidean', document_metric=None, matrix=True):
    """
    Calculate distances between documents by calculating distance between sentences.

    parameters:
    -----------
    annotation: pd.DataFrame
        The result of an annotation

    metric: None | String | 'poscorr'
        The distance metric to use between sentences. See fastcluster.pdist
        poscorr is positive correlation

    merge_dist: None | min | max | mean | median
        How to calculate distances between documents, on the basis of the distances between sentences

    Output
    ------
    Y: ndarray
        Returns a condensed distance matrix Y.  For
        each :math:`i` and :math:`j` (where :math:`i<j<m`),where m is the number
        of original observations. The metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry ``ij``.
    """
    #Remove sentences with no annotations
    validate(A, False)

    O_doc_idx = A.document_.copy()
    N_docs = A.document_.unique().shape[0]
    A = A[_features(A).sum(axis=1) > 0]
    D_doc_idx = A.document_

    if metric == 'poscorr':
        D_S = np.corrcoef(_features(A).values)
        D_S[D_S < 0] = 0
        D_S = 1 - D_S
    else:
        D_S = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(_features(A), metric=metric))
    #fi
    
    document_metric = {'min':np.min, 'max':np.max,
                       'mean':np.mean, None: np.min,
                       'median':np.median}[document_metric]

    D_D = np.zeros((N_docs, N_docs))
    for i in range(N_docs):
        for j in range(i, N_docs):
            if i == j:
                D_D[i,j] = 0
            else:
                rm = D_S[D_doc_idx == i,:][:,D_doc_idx == j]
                D_D[i,j] = document_metric(rm) if np.prod(rm.shape) > 0 else np.inf
                D_D[j,i] = D_D[i,j]
            #fi
        #efor
    #efor
    return D_D if matrix else scipy.spatial.distance.squareform(D_D)
#edef

#########################################################################

def pagerank(D, eps=1.0e-8, d=0.85):
    """
    The PageRank Algorithm (taken from wikipedia)

    Parameters:
    -----------
    D: pd.DataFrame
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1

    d: Float
        damping factor (default value 0.85)

    eps: Float
        quadratic error for v (default value 1.0e-8)

    Output:
    -------
        A vector of ranks such that v_i is the i-th rank from [0, 1]
    """
    
    N = D.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    last_v = np.ones((N, 1), dtype=np.float32) * 100

    while np.linalg.norm(v - last_v, 2) > eps:
        last_v = v
        v = d * np.matmul(D, v) + (1 - d) / N
    #ewhile
    return v
#edef

#########################################################################