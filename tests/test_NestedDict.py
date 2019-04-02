from resources.NestedDict import NestedDict
import unittest

import sys
sys.path.append(
    '/home/steven/Documents/Projects/Python/NLP/sentiment_analysis/')


class TestNestedDict(unittest.TestCase):

    def test_getitem(self):
        testdict = NestedDict({'b': {'c': 50}})
        self.assertEqual(testdict['b']['c'], 50)
        self.assertEqual(testdict['b'], NestedDict({'c': 50}))
        self.assertDictEqual(testdict['d'], NestedDict())

    def test_get_by_path(self):
        testdict = NestedDict({'a': {'b': {'c': 100}}, 'd': 1000})
        self.assertEqual(testdict.get_by_path(['a', 'b', 'c']), 100)

    def test_set_by_path(self):
        testdict = NestedDict({'d': 1000})
        testdict.set_by_path(['a', 'b', 'c'], 100)
        self.assertEqual(testdict.get_by_path(['a', 'b', 'c']), 100)

    def test_add_by_path(self):
        testdict = NestedDict({'d': 1000})
        testdict.add_by_path(['a', 'b', 'c'], 100)
        self.assertEqual(testdict.get_by_path(['a', 'b', 'c']), 100)
        testdict.add_by_path(['a', 'b', 'c'], 100)
        self.assertEqual(testdict.get_by_path(['a', 'b', 'c']), 200)


if __name__ == '__main__':
    unittest.main()
