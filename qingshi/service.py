# -*- coding: utf8 -*-
import logging

import redis

from qingshi import lexicon

class SegumentService(object):

    def __init__(self, config, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self.ngram = 2
        self.config = config

        # get ngram configuration
        c = config.get('lexicon')
        if c:
            self.ngram = c.get('ngram', self.ngram)
            self.logger.info('ngram %d', self.ngram)

        # get redis config
        c = config.get('redis', {})
        redis_db = redis.Redis(**c)

        self.db = lexicon.LexiconDatabase(redis_db, self.ngram)
        self.builder = lexicon.LexiconBuilder(self.db, self.ngram)

    def getStats(self):

        return self.db.getStats()

    def feed(self, category, text):
 
        self.logger.info('Feed %d bytes data', len(text))
        self.logger.info('ngram %d', self.ngram)
        return self.builder.feed(category, text)

    def splitTerms(self, text, categories=None):

        terms = []
        for sentence in lexicon.splitSentence(text):
            if sentence:
                for mixed in lexicon.iterMixTerms(sentence):
                    # English term
                    if mixed.startswith('E'):
                        terms.append(mixed)
                    # Chinese sentence
                    else:
                        terms.extend(self.db.splitTerms(mixed, categories))
        return terms

    def splitNgramTerms(self, text):

        terms = []
        for sentence in lexicon.splitSentence(text):
            if sentence:
                for mixed in lexicon.iterMixTerms(sentence):
                    # English term
                    if mixed.startswith('E'):
                        terms.append(mixed)
                    # Chinese sentence
                    else:
                        for n in xrange(1, self.ngram+1):
                            terms.extend(lexicon.iterTerms(n, mixed, False))
        return terms

    def splitSentence(self, text):

        return lexicon.splitSentence(text)

    def splitMixTerms(self, text):

        return list(lexicon.iterMixTerms(text))
