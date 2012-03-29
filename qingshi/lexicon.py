# -*- coding: utf8 -*-

import re
import logging


# default delimiters for splitSentence
# # is the reserved word in the system
default_delimiters = set(u"""#\n\r\t ,.:?!"()[]{}。，、；：？！“”「」『』《》〈〉【】〖〗〔〕«»─（）﹝﹞…﹏＿‧""")

eng_term_pattern = """[a-zA-Z0-9\\-_']+"""

def splitNgrams(n, terms):

    for i in xrange(len(terms) - n + 1):
        yield terms[i:i+n]

def iterEnglishTerms(text):

    terms = []
    parts = text.split()
    for part in parts:
        for term in re.finditer(eng_term_pattern, part):
            terms.append(term.group(0))
    return terms

def iterMixTerms(text, eng_prefix='E'):
    # last position term
    terms = []
    parts = text.split()
    for part in parts:
        last = 0
        for match in re.finditer(eng_term_pattern, part):
            previous_term = part[last:match.start()]
            if previous_term:
                terms.append(previous_term)
            if eng_prefix:
                terms.append(eng_prefix + match.group(0).lower())
            else:
                terms.append(match.group(0).lower())
            last = match.end()
        final_term = part[last:]
        if final_term:
            terms.append(final_term)
    return terms

def splitSentence(text, delimiters=None):

    if delimiters is None:
        delimiters = default_delimiters

    sentence = []
    for c in text:
        if c in delimiters:
            yield ''.join(sentence)
            sentence = []
        else:
            sentence.append(c)
    yield ''.join(sentence)

def iterSentenceRela2(maxn, text):

    for sentence in splitSentence(text):
        terms = []
        # 0 < s1 < e1 < s2 < e2 < len
        # term = (s, e). And s is closed, e is opend
        ext = len(sentence)
        for s1 in xrange(ext):
            for e1 in xrange(s1+1, min(s1+maxn+1, ext+1)):
                for s2 in xrange(e1, ext):
                    for e2 in xrange(s2+1, min(s2+maxn+1, ext+1)):
                        terms.append((sentence[s1:e1], sentence[s2:e2]))
        yield terms


def iterSentenceTerms(maxn,text):

    for sentence in splitSentence(text):
        terms = []
        for n in xrange(1, maxn+1):
            terms.extend(splitNgrams(n, sentence))
        yield terms

def iterTerms(n, text, emmit_head_tail=False):
    
    for sentence in splitSentence(text):
        first = True
        term = None
        for term in splitNgrams(n, sentence):
            term = term.lower()
            yield term
            if first:
                if emmit_head_tail:
                    yield 'B' + term
                first = False
        if term is not None:
            if emmit_head_tail:
                yield 'E' + term

def findBestSegment(grams, op=lambda a, b: a*b):

    def makeTuple(term):
        return ([term[0]], term[1])
    # n-gram
    n = len(grams)
    # size of terms in unigram
    size = len(grams[0])

    # table for best solution in range (begin, end)
    table = {}

    # initialize the best solution table
    for i in xrange(size):
        term = grams[0][i]
        table[(i, i)] = makeTuple(term)

    # get candidate in of (left items, right items) at i-th item
    def getCandidate(i, left, right):
        left_range = (i, i+left-1)
        right_range = (i+left, i+left+right-1)
        left_item = table[left_range]
        right_item = table[right_range]
        return (left_item[0]+right_item[0], op(left_item[1], right_item[1]))

    # handle current_size together cases, for example
    # e.g. C1,C2,C3,C4, if the current_size is 2, then the cases will be
    # [C1, C2], [C2, C3], [C3, C4]
    for current_size in xrange(2, size+1):
        for i in xrange(size - current_size + 1):
            # handle all possible partition cases,
            # e.g. with current_size 4, we can have partition cases:
            # (1, 3), (3, 1), (2, 2)
            candidates = []
            for count in xrange(1, (current_size/2) + 1):
                # count of left and right partition
                left, right = count, current_size - count
                candidates.append(getCandidate(i, left, right))
                if left != right:
                    left, right = right, left
                    candidates.append(getCandidate(i, left, right))
            if current_size <= n:
                candidates.append(makeTuple(grams[current_size-1][i]))
            # sort candidates
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            winner = candidates[0]
            current_range = (i, i+current_size-1)
            table[current_range] = winner
    return table[(0, size-1)]

class LexiconCategory(object):

    progress_interval = 10000

    def __init__(self, db, name, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger('lexicon.database')
        self.db = db
        # name of category
        self.name = name
        assert ':' not in self.name, """ ":" can't be part of category name"""

        # prefix of this category
        self.prefix = db.prefix + self.name + ':'
        self._meta_prefix = self.prefix + 'meta:'
        self._lexicon_prefix = self.prefix + 'lex:'
        self._terms_key = self.prefix + 'terms'
        self._relation_prefix = self.prefix + 'rel:'
        self._relaref_prefix = self.prefix + 'ref:'
        self._relas_key = self.prefix + 'relas'


    def init(self, ngram):
        """Initialize category in database

        """
        # add to category set
        if not self.db.redis.sadd(self.db._category_set_key, self.name):
            # already exists
            self.logger.info('Category %s already exists', self.name)
            return
        self.setMeta('gram', ngram)
        for n in xrange(ngram):
            self.increaseGramSum(n, 0)
            self.increaseGramVariety(n, 0)
        self.logger.info('Add category %s (gram=%s)', self.name, ngram)


    def clean(self):
        """Clean all value of this category

        """
        # remove terms
        terms = self.getTermList()
        keys = [self._lexicon_prefix + term for term in terms]
        if keys:
            self.db.redis.delete(*keys)
        # remove link node
        keys = [self._lexicon_prefix + 'lhs:' + term for term in terms]
        if keys:
            self.db.redis.delete(*keys)
        keys = [self._lexicon_prefix + 'rhs:' + term for term in terms]
        if keys:
            self.db.redis.delete(*keys)

        # remove relations
        rels = self.getRelaList()
        keys = [self._relation_prefix + rel for rel in rels]
        if keys:
            self.db.redis.delete(*keys)
        # remove relation refined
        keys = [self._relaref_prefix + rel for rel in rels]
        if keys:
            self.db.redis.delete(*keys)

        # remove meta keys
        for n in xrange(0, self.gram + 1):
            self.db.redis.delete(self._meta_prefix + ('%s-gram-sum' % n))
            self.db.redis.delete(self._meta_prefix + ('%s-gram-variety' % n))
        self.db.redis.delete(self._meta_prefix + 'gram')


        # remove this category from category set
        self.db.redis.srem(self.db._category_set_key, self.name)
        self.db.redis.delete(self._terms_key)
        self.db.redis.delete(self._relas_key)

        self.logger.info('Clean category %r, %d terms are deleted',
                         self.name, len(terms))

    def getMeta(self, key):
        """Get value of a meta data

        """
        return self.db.redis.get(self._meta_prefix + key)

    def setMeta(self, key, value):
        """Set value of a meta data

        """
        return self.db.redis.set(self._meta_prefix + key, value)

    @property
    def gram(self):
        return int(self.getMeta('gram') or 0)

    def increaseTerm(self, term, delta=1):
        """Increase value of a term

        """
        # increase number
        key = self._lexicon_prefix + term
        self.db.redis.incr(key, delta)
        # add to terms set
        self.db.redis.sadd(self._terms_key, term)

    def increaseRela(self, rel, delta=1):
        """Increase value of a term

        """
        # increase number
        key = self._relation_prefix + rel
        self.db.redis.incr(key, delta)
        # add to terms set
        self.db.redis.sadd(self._relas_key, rel)

    def getTerm(self, term):
        """Get count of a term

        """
        key = self._lexicon_prefix + term
        return self.db.redis.get(key)

    def getTerms(self, *terms):
        """Get count of terms

        """
        keys = [self._lexicon_prefix + term for term in terms]
        return self.db.redis.mget(keys)

    def getTermList(self):
        """Get all term name in this category

        """
        return self.db.redis.smembers(self._terms_key)

    def getSortTermList(self):
        """Get all term name in this category, sorted by counting of the term

        """
        return self.db.redis.sort(self._terms_key, start=0, num=100,
            by=self._lexicon_prefix + '*', desc=True)

    # Add for relations
    def getRelaList(self):
        """Get all relations name in this category

        """
        return self.db.redis.smembers(self._relas_key)

    def getRela(self, rel):
        """Get count of a relation

        """
        key = self._relation_prefix + rel
        return self.db.redis.get(key)

    def getRelas(self, *rels):
        """Get count of relations

        """
        keys = [self._relation_prefix + rel for rel in rels]
        return self.db.redis.mget(keys)


    def getSortRelaList(self):
        """Get all relations name in this category, sorted by counting of the relation

        """
        return self.db.redis.sort(self._relas_key, start=0, num=100,
            by=self._relation_prefix + '*', desc=True)

    def getRelaRef(self, rel):
        """Get count of a relation

        """
        key = self._relaref_prefix + rel
        return self.db.redis.get(key)

    def getRelaRefs(self, *rels):
        """Get count of relations

        """
        keys = [self._relaref_prefix + rel for rel in rels]
        return self.db.redis.mget(keys)

    def getSortRelaRefList(self):
        """Get all relations name in this category, sorted by counting of the relation

        """
        return self.db.redis.sort(self._relas_key, start=0, num=100,
            by=self._relaref_prefix + '*', desc=True)

    def increaseGramSum(self, n, value):
        """Increase sum of n-gram terms

        """
        key = self._meta_prefix + ('%s-gram-sum' % n)
        return self.db.redis.incr(key, value)

    def increaseGramVariety(self, n, value):
        """Increase variety of n-gram terms

        """
        key = self._meta_prefix + ('%s-gram-variety' % n)
        return self.db.redis.incr(key, value)

    def getGramSum(self, n):
        """Get sum of n-gram terms

        """
        key = '%s-gram-sum' % n
        return int(self.getMeta(key) or 0)

    def getGramVariety(self, n):
        """Get variety of n-gram terms

        """
        key = '%s-gram-variety' % n
        return int(self.getMeta(key) or 0)

    # Add for post process
    def setLink(self, srckey, pattern):
        """Get srckey as link node containg related keys which matching pattern
        """
        destkeys = self.db.redis.keys(pattern)
        for destkey in destkeys:
            self.db.redis.zadd(srckey, destkey, self.db.redis.get(destkey))

    def setRelaProb(self, rel):
        """ Calculate the probability of relations according to all related words
        """
        terms = rel.split('#')
        values = self.getTerms(*terms)

        comb = float(self.getRela(rel))
        mult = float(1)
        for value in values:
            mult = mult * float(value)
        newkey = self._relaref_prefix + rel
        self.db.redis.set(newkey, comb/mult)

    def refine(self):
        """Refine all terms of this category by building their connections

        """
#        terms = self.getTermList()
#        for term in terms:
#            lhsext = self._lexicon_prefix + '???' + term
#            rhsext = self._lexicon_prefix + term + '???'
#            lhskey = self._lexicon_prefix + 'lhs:' + term
#            rhskey = self._lexicon_prefix + 'rhs:' + term
#            self.setLink(lhskey, lhsext)
#            self.setLink(rhskey, rhsext)
#
        rels = self.getRelaList()
        for rel in rels:
            self.setRelaProb(rel)


    #
    def getStats(self):
        """Get statistics of this category
        """
        stats = dict(
            gram=self.gram,
            total_sum=0,
            total_variety=0
        )
        for n in xrange(1, self.gram + 1):
            sum = self.getGramSum(n)
            variety = self.getGramVariety(n)
            sum_key = '%sgram_sum' % n
            variety_key = '%sgram_variety' % n
            stats[sum_key] = sum
            stats[variety_key] = variety
            stats['total_sum'] += sum
            stats['total_variety'] += variety
        return stats

    def dump(self, file, level):
        self.logger.info('Dumping meta-data ...')
        print >>file, 'gram', self.gram
        for n in xrange(1, self.gram + 1):
            name = '%d-gram-sum' % n
            value = self.getGramSum(n)
            print >>file, name, value
            self.logger.info('Meta-data %s=%s', name, value)
            name = '%d-gram-variety' % n
            value = self.getGramVariety(n)
            print >>file, name, value
            self.logger.info('Meta-data %s=%s', name, value)

        if (level == 'detail'):
            # a blank line
            print >>file

            self.logger.info('Dumping lexicons terms ...')
            terms = self.getTermList()
            self.logger.info('Get %d terms', len(terms))
            self.logger.info('Dumping lexicons values ...')
            values = self.getTerms(*terms)
            self.logger.info('Get %d values', len(values))
            for i, (term, count) in enumerate(zip(terms, values)):
                term = term.decode('utf8')
                print >>file, count, term
                if i % self.progress_interval == 0:
                    whole = len(terms)
                    per = (i/float(whole))*100.0
                    self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)

        # a blank line
        print >>file

        self.logger.info('Dumping lexicons sorted terms ...')
        terms = self.getSortTermList()
        self.logger.info('Get %d terms', len(terms))
        self.logger.info('Dumping lexicons values ...')
        values = self.getTerms(*terms)
        self.logger.info('Get %d values', len(values))
        for i, (term, count) in enumerate(zip(terms, values)):
            lhskey = self._lexicon_prefix + 'lhs:' + term
            rhskey = self._lexicon_prefix + 'rhs:' + term

            term = term.decode('utf8')
            print >>file, count, term

            #  zrevrange(self, name, start, num, withscores=False, score_cast_func=<type 'float'>)
            print >>file, "\t", lhskey.decode('utf8')
            links = self.db.redis.zrevrange(lhskey, 0, -1, withscores=True)
            for j, (extterm, extcount) in enumerate(links):
                extterm = extterm.decode('utf8')
                print >>file, "\t\t", extcount, extterm

            print >>file, "\t", rhskey.decode('utf8')
            links = self.db.redis.zrevrange(rhskey, 0, -1, withscores=True)
            for j, (extterm, extcount) in enumerate(links):
                extterm = extterm.decode('utf8')
                print >>file, "\t\t", extcount, extterm

            if i % self.progress_interval == 0:
                whole = len(terms)
                per = (i/float(whole))*100.0
                self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)

        # a blank line
        print >>file

        self.logger.info('Dumping lexicons sorted relations ...')
        rels = self.getSortRelaList()
        self.logger.info('Get %d relations', len(rels))
        self.logger.info('Dumping relations values ...')
        values = self.getRelas(*rels)
        self.logger.info('Get %d values', len(values))
        for i, (rel, count) in enumerate(zip(rels, values)):
            rel = rel.decode('utf8')
            print >>file, count, rel
            if i % self.progress_interval == 0:
                whole = len(rels)
                per = (i/float(whole))*100.0
                self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)

        # a blank line
        print >>file

        self.logger.info('Dumping lexicons sorted ref-relations ...')
        rels = self.getSortRelaRefList()
        self.logger.info('Get %d relations', len(rels))
        self.logger.info('Dumping relations values ...')
        values = self.getRelaRefs(*rels)
        self.logger.info('Get %d values', len(values))
        for i, (rel, count) in enumerate(zip(rels, values)):
            rel = rel.decode('utf8')
            print >>file, count, rel
            if i % self.progress_interval == 0:
                whole = len(rels)
                per = (i/float(whole))*100.0
                self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)


class LexiconDatabase(object):

    progress_interval = 10000

    def __init__(
        self,
        redis,
        ngram,
        prefix='qingshi:',
        logger=None
    ):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger('lexicon.database')
        self.redis = redis
        self.ngram = ngram
        self.prefix = prefix

        self._categories_cache = {}
        # key for category
        self._category_set_key = self.prefix + 'category'

    def getCategory(self, name):
 
        if name not in self.getCategoryList():
            return
        category = self._categories_cache.get(name)
        if category:
            return category
        category = LexiconCategory(self, name)
        self._categories_cache[name] = category
        return category

    def addCategory(self, name):

        category = self._categories_cache.get(name)
        if category:
            return category
        category = LexiconCategory(self, name)
        category.init(self.ngram)
        self._categories_cache[name] = category
        return category

    def getCategoryList(self):

        return self.redis.smembers(self._category_set_key)

    def clean(self, categories=None):

        all_category = self.getCategoryList()
        if not categories:
            categories = all_category

        if categories:
            for name in categories:
                c = self.getCategory(name)
                if c:
                    c.clean()
        self.logger.info('Clean lexicon database, %s categories',
                         len(categories))

    def _getTermScore(self, term, ngram, categories):

        score = 0.00000001
        for c in categories:
            count = int(c.getTerm(term) or 0)
            n = len(term)
            sum = int(c.getGramSum(n) or 0)
            variety = int(c.getGramVariety(n) or 0)
            if not variety:
                v = 1
            else:
                v = sum/float(variety)
                v *= v
            score += count/v
        return score

    def splitTerms(self, text, categories=None):

        all_category = self.getCategoryList()
        if not categories:
            categories = all_category
        c_list = []
        for name in categories:
            c = self.getCategory(name)
            if not c:
                self.logger.error('Category %s not exist', name)
                continue
            c_list.append(c)
        grams = []
        for n in xrange(1, self.ngram+1):
            terms = []
            for term in splitNgrams(n, text):
                score = self._getTermScore(term, n, c_list)
                self.logger.debug('Term=%s, Score=%s', term, score)
                terms.append((term, score))
            grams.append(terms)
        terms, best_score = findBestSegment(grams)
        self.logger.debug('Best score: %s', best_score)
        return terms

class LexiconBuilder(object):

    progress_interval = 10000

    def __init__(self, db, ngram, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger('lexicon.builder')
        self.db = db
        self.ngram = ngram

    def feed(self, category, text):

        cat = self.db.addCategory(category)
        total = 0
        # feed term by term
#        for n in xrange(1, self.ngram+1):
#            self.logger.debug('Processing %d-gram', n)
#            terms_count = {}
#            sum = 0
#            variety = 0
#            # count number of terms
#            for term in iterTerms(n, text):
#                terms_count.setdefault(term, 0)
#                if terms_count[term] == 0:
#                    variety += 1
#                terms_count[term] += 1
#                total += 1
#            # add terms to database
#            for i, (term, delta) in enumerate(terms_count.iteritems()):
#                result = cat.increaseTerm(term, delta)
#                sum += delta
#                if i % self.progress_interval == 0:
#                    whole = len(terms_count)
#                    per = (i/float(whole))*100.0
#                    self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)
#
#            # add n-gram count
#            result = cat.increaseGramSum(n, sum)
#            self.logger.debug('Increase %d-gram sum to %d', n, result)
#            result = cat.increaseGramVariety(n, variety)
#            self.logger.debug('Increase %d-gram variety to %d', n, result)


        # feed sentence by sentences with relations
        self.logger.debug('Processing relations')
        rels_count= {}
        rels_sum = 0
        rels_variety = 0
        rels_total = 0
        for rels in iterSentenceRela2(self.ngram, text):
            for rel in rels:
                rel = rel[0] + '#' + rel[1]
                rels_count.setdefault(rel, 0)
                if rels_count[rel] == 0:
                    rels_variety += 1
                rels_count[rel] += 1
                rels_total += 1
        for i, (rel, delta) in enumerate(rels_count.iteritems()):
             #print "Rela:", i, rel, delta
             #result = cat.increaseRela(rel, delta)
             rels_sum += delta
             if i % self.progress_interval == 0:
                 whole = len(rels_count)
                 per = (i/float(whole))*100.0
                 self.logger.info('Progress %d/%d (%02d%%)', i, whole, per)


        self.logger.info('Fed %d terms', total)
        return total
