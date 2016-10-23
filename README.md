> There should be one-- and preferably only one --obvious way to do it.

> *Tim Peters, The Zen of Python (PEP 20)*

[The Elements of Python Style](https://github.com/amontalenti/elements-of-python-style)

Reverse a list
```python
l = [0,1,2,3,4,5]
# Create a new copy with the elements reversed (slicing).
rev = l[::-1]

# Get an iterator that walks the original list in reverse order.
rev = reversed(l)

# Reverse in place.
l.reverse()
```

Add element(s) to a list.
```python
# Append an item to the end.
a = [1, 2, 3]
a.append([4, 5])
print a
# [1, 2, 3, [4, 5]]

# Extend the list by appending items from iterable.
a = [1, 2, 3]
a.extend([4, 5]) # same as: a += [4, 5]
print a
# [1, 2, 3, 4, 5]

# Insert a new item before a given position.
a = [1, 2, 3]
a.insert(1, '?')
print a
# [1, '?', 2, 3]
```

Remove element from a list.
```python
# To remove an element's first occurrence in a list:
a = ['a', 'b', 'c', 'd']
a.remove('b')
print a
# ['a', 'c', 'd']

# Remove all occurences of an element in a list.
a = [1, 2, 3, 4, 2, 3, 4, 2, 7, 2]
a = [x for x in a if x != 2]
print a
# [1, 3, 4, 3, 4, 7]

# Remove element by index.
a = ['a', 'b', 'c', 'd']
a.pop(1)
# 'b'
print a
['a', 'c', 'd']
```

Find element in a list
```python
# Find the index of the first item found. Raises an error if item is not found.
a = ['a', 'b', 'c', 'd', 'b']
print a.index('b')
# 1
print a.index('x')
# ValueError: 'x' is not in list

# Find all occurences of an item.
a = ['a', 'b', 'c', 'd', 'b']
indexes = [i for i, item in enumerate(a) if item == 'b']
# [1, 4]

# Bad:
i = a.index('x') if 'x' in a else None
print i
# None

# Good:
try:
    i = a.index('b')
    # The code below can assume that `i` is a valid index.
    return i
except ValueError:
    # We get a chance to handle the error.
    return None
```

List slice assignment
```python
a = [1, 2, 3, 4, 5]
a[2:3] = [0, 0]
print a
# [1, 2, 0, 0, 4, 5]
a[1:1] = [8, 9]
print a
# [1, 8, 9, 2, 0, 0, 4, 5]
a[1:-1] = []
print a
# [1, 5]
```

Get all even numbers from a list located at even indexes
```python
[x for x in arr[::2] if x%2 == 0]
```

Get the first item from an iterable matching a condition
```python
print next((x for x in (1,2,3) if x > 1), None)
# 2
print next((x for x in (1,2,3) if x > 3), None)
# None
```

Simplify Chained Comparison
```python
x < y <= z # equivalent to: x < y and y <= z
```

Zipping and Unzipping
```python
a = [1, 2, 3]
b = ['a', 'b', 'c']
z = zip(a, b)
print z
# [(1, 'a'), (2, 'b'), (3, 'c')]

zip(*z)
# [(1, 2, 3), ('a', 'b', 'c')]

a = (1, 2, 3)
b = ('a', 'b', 'c')
zip(*zip(a, b)) == [a, b]
# True
```

Sliding windows (n-grams) using zip and iterators
```python
from itertools import islice
def n_grams(a, n):
	z = (islice(a, i, None) for i in range(n))
	return zip(*z)

a = [1, 2, 3, 4, 5, 6]
n_grams(a, 3)
# [(1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
n_grams(a, 2)
# [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
n_grams(a, 4)
# [(1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)]
```

Timeit
```python
import timeit

setup = 'import random;a = [random.random() for _ in xrange(100000)]'
n = 1000

print timeit.timeit('b=reversed(a)', setup, number=n) # 0.0002
print timeit.timeit('a.reverse()', setup, number=n)   # 0.0600
print timeit.timeit('b=a[::-1]', setup, number=n)     # 0.7700
```

Counter
```python
colors = ['red', 'green', 'red', 'blue', 'green', 'red']

d = {}
for color in colors:
    d[color] = d.get(color, 0) + 1
# {'blue': 1, 'green': 2, 'red': 3}
```

Grouping with dictionaries
```python
from collections import defaultdict

# Group names by name length
names = ['raymond', 'rachel', 'matthew', 'roger',
         'betty', 'melissa', 'judith', 'charlie']

d = defaultdict(list)
for name in names:
    key = len(name)
    d[key].append(name)

# {5: ['roger', 'betty'], 6: ['rachel', 'judith'], 7: ['raymond', 'matthew', 'melissa', 'charlie']}
```

Sort characters in a string by frequency
```python
from collections import Counter
str = 'Mississippi'
dict = Counter(str)
sorted(dict.items(), key=lambda x: -x[1])
# [('i', 4), ('s', 4), ('p', 2), ('M', 1)]
```

```python
str = 'Mississippi'
sorted((-str.count(w), w) for w in set(str))
# [(-4, 'i'), (-4, 's'), (-2, 'p'), (-1, 'M')]
```

Sort letters from file by frequency
```python
from collections import Counter
with open('alice.txt') as f:
    text = f.read().lower()
    letters = (c for c in text if c.isalpha())
    sorted(Counter(letters).items(), key=lambda x: -x[1])
```

Remove all spaces from string
```python
str = str.replace(' ', '')
```

Remove all whitespace characters (space, tab, newline, and so on)
```python
str = ''.join(str.split())
```

Split text
```python
text = "Hi, John, this is me - Paul. What's up man?"

text.split()
# ['Hi,', 'John,', 'this', 'is', 'me', '-', 'Paul.', "What's", 'up', 'man?']

import re
re.split('\W+', text)
# ['Hi', 'John', 'this', 'is', 'me', 'Paul', 'What', 's', 'up', 'man', '']
```

Sort words in text by frequency
```python
import re
from collections import Counter
text = open('alice.txt').read().lower()
words = re.split('\W+', text)
sorted_by_freq = Counter(words).most_common()
```

Iterators for efficient looping
https://docs.python.org/2.7/library/itertools.html

```python
from itertools import combinations
list(''.join(comb) for comb in combinations('ABCD', 3))
# ['ABC', 'ABD', 'ACD', 'BCD']
```

| Function | Description | Example |
| ---------| ----------- | ------- |
| count(start, [step]) | start, start+step, start+2*step, ... | count(10) # 10 11 12 13 14 ... |
| cycle(p) | p0, p1, ... plast, p0, p1, ... | cycle('ABCD') # A B C D A B C D ... |
| repeat(elem [,n]) | elem, elem, elem, ... endlessly or up to n times | repeat(10, 3) # 10 10 10 |
| chain(p, q, ...) | p0, p1, ... plast, q0, q1, ... | chain('ABC', 'DEF') # A B C D E F |
| compress(data, selectors) | (d[0] if s[0]), (d[1] if s[1]), ... | compress('ABCDEF', [1,0,1,0,1,1]) # A C E F |
| dropwhile(pred, seq) | seq[n], seq[n+1], starting when pred fails | dropwhile(lambda x: x<5, [1,4,6,4,1]) # 6 4 1 |
| groupby(iterable[, keyfunc]) | sub-iterators grouped by value of keyfunc(v) |   |
| ifilter(pred, seq) | elements of seq where pred(elem) is true | ifilter(lambda x: x%2, range(10)) # 1 3 5 7 9 |
| ifilterfalse(pred, seq) | elements of seq where pred(elem) is false | ifilterfalse(lambda x: x%2, range(10)) # 0 2 4 6 8 |
| islice(seq, [start,] stop [, step]) | elements from seq[start:stop:step] | islice('ABCDEFG', 2, None) # C D E F G |
| imap(func, p, q, ...) | func(p0, q0), func(p1, q1), ... | imap(pow, (2,3,10), (5,2,3)) # 32 9 1000 |
| starmap(func, seq) | func(*seq[0]), func(*seq[1]), ... | starmap(pow, [(2,5), (3,2), (10,3)]) # 32 9 1000 |
| tee(it, n) | it1, it2, ... itn splits one iterator into n |   |
| takewhile(pred, seq) | seq[0], seq[1], until pred fails | takewhile(lambda x: x<5, [1,4,6,4,1]) # 1 4 |
| izip(p, q, ...) | (p[0], q[0]), (p[1], q[1]), ... | izip('ABCD', 'xy') # Ax By |
| izip_longest(p, q, ...) | (p[0], q[0]), (p[1], q[1]), ... | izip_longest('ABCD', 'xy', fillvalue='-') # Ax By C-D- |
| product(p, q, ... [repeat=1]) | cartesian product, equivalent to a nested for-loop | product('xyz', '12') # [('x', '1'), ('x', '2'), ('y', '1'), ('y', '2'), ('z', '1'), ('z', '2')] |
| permutations(p[, r]) | r-length tuples, all possible orderings, no repeated elements | permutations('ABCD', 2) # AB AC AD BA BC BD CA CB CD DA DB DC |
| combinations(p, r) | r-length tuples, in sorted order, no repeated elements | combinations('ABCD', 2) # AB AC AD BC BD CD |
| combinations_with_replacement(p, r) | r-length tuples, in sorted order, with repeated elements | combinations_with_replacement('ABCD', 2) # AA AB AC AD BB BC BD CC CD DD |

Create a deck of cards
```python
from collections import namedtuple
from itertools import product

Card = namedtuple('Card', 'rank suit')
RANKS = '23456789TJQKA'
SUITS = ('Clubs', 'Diamonds', 'Hearts', 'Spades')
DECK = tuple(Card(*card) for card in product(RANKS, SUITS))
```

Interpose
```python
from itertools import islice, chain, repeat

def interpose(seq, sep):
    """Introduce sep between each pair of elements in seq."""
    return islice(chain.from_iterable(zip(repeat(sep), seq)), 1, None)

list(interpose([1, 2, 3], '-'))
# [1, '-', 2, '-', 3]

def interpose(seq, sep):
    result = []
    for x in seq:
        result.extend([x, sep])
    return result[:-1]

interpose([1, 2, 3], '-')
# [1, '-', 2, '-', 3]
```


Classes
```python
class Person(object):
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        self.__name = value

    def say_name(self):
        print('My name is {}'.format(self.__name))

p = Person('John')
p.say_name()
```

Raise and catch exception
```python
def sqrt(x):
    if x < 0:
        raise ValueError('x must be greater than 0, got {}'.format(x))
    return x**0.5

try:
    print(sqrt(-1))
except ValueError as e:
    print('Error: {}'.format(e))
else:
    print('Strange, sqrt(-1) did not raise and error!')
```

Scrape all links from a web page
```python
import requests
from bs4 import BeautifulSoup

URL = 'https://www.tradeo.com/'
html = requests.get(URL).text
doc = BeautifulSoup(html)
links = (element.get('href') for element in doc.find_all('a'))
print('\n'.join(sorted(links)))
```

Split array into chunks
```python
# Encode the message (convert characters ASCII codes to binary)
message = 'c omp   re hensions'
encoded = ''.join('{0:08b}'.format(ord(ch)) for ch in message)
# '011000110010000001101111011011000001101000011001010110111001110011..."

# Decode the message (split into octets and convert to letters)
octets = [encoded[i:i+8] for i in range(0, len(encoded), 8)]
chrs = [chr(int(octet, 2)) for octet in octets]
chrs_no_spaces = [c for c in chrs if c != ' ']
decoded = ''.join(chrs_no_spaces)
print(decoded)
# comprehensions
```

Dict Comprehensions
```python
d = {i : chr(65+i) for i in range(4)}
# {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

d_inverted = {v:k for k, v in d.items()}
# {'A': 0, 'C': 2, 'B': 1, 'D': 3}

ships = [
  {'id': 0, 'model': 'T-65B X-wing'},
  {'id': 1, 'model': 'TIE Advanced x1'},
]
ships_by_id = {s['id']:s['model'] for s in ships}
# {0: 'T-65B X-wing', 1: 'TIE Advanced x1'}

x = (1,'a',2,'b',3,'c')
dict(x[i:i+2] for i in range(0, len(x), 2))
# {1: 'a', 2: 'b', 3: 'c'}
```

List Flatten (with nested comprehension)
```python
[x for sublist in [[1,2],[3,4],[5,6,7]] for x in sublist]
# [1, 2, 3, 4, 5, 6, 7]

a = [1, 2, [3, 4], [[5, 6], [7, 8]]]
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
flatten(a)
# [1, 2, 3, 4, 5, 6, 7, 8]
>>>
```

Set Comprehension
```python
# Find the planets from all episodes of Star Wars
episodes = {
  'Episode I': {'planets': ['Naboo', 'Tatooine', 'Coruscant']},
  'Episode II': {'planets': ['Geonosis', 'Kamino', 'Geonosis']},
  'Episode III': {'planets': ['Felucia', 'Utapau', 'Coruscant', 'Mustafar']},
  'Episode IV': {'planets': ['Tatooine', 'Alderaan', 'Yavin 4']},
  'Episode V': {'planets': ['Hoth', 'Dagobah', 'Bespin']},
  'Episode VI': {'planets': ['Tatooine', 'Endor']},
  'Episode VII': {'planets': ['Jakku', 'Takodana', 'Ahch-To']},
}

# from itertools import chain
# planets = set(chain.from_iterable(e['planets'] for e in episodes.values()))
planets = {p for e in episodes.values() for p in e['planets']}
print('There are {} planets: {}'.format(len(planets), ', '.join(planets)))
# There are 17 planets: Utapau, Ahch-To, Kamino, Bespin, Naboo, Jakku, Felucia, Tatooine, Mustafar, Takodana, Yavin 4, Alderaan, Endor, Dagobah, Hoth, Coruscant, Geonosis
```

Docstrings (see [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html))
```python
def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    return True
```

Fibonacci
```python
def fibonacci(n):
    x, y = 0, 1
    for i in range(n):
        print x
        x, y = y, x + y
```

Implement `itertools.islice(iterable, start, stop[, step])`
```python
def islice(iterable, *args):
	"""Return an iterator that returns selected elements from the iterable."""
    s = slice(*args)
    it = iter(xrange(s.start or 0, s.stop or sys.maxint, s.step or 1))
    nexti = next(it)
    for i, element in enumerate(iterable):
        if i == nexti:
            yield element
            nexti = next(it)

# islice('ABCDEFG', 2) --> A B
# islice('ABCDEFG', 2, 4) --> C D
# islice('ABCDEFG', 2, None) --> C D E F G
# islice('ABCDEFG', 0, None, 2) --> A C E G
```

Deque (double-ended queue)
```python
from collections import deque
q = deque()
q.append(1)
q.appendleft(2)
q.extend([3, 4])
q.extendleft([5, 6])
q
# deque([6, 5, 2, 1, 3, 4])
q.pop()
# 4
q.popleft()
# 6
```

Print a comma separated list of numbers
```python
a = ['a','b','c','d', 4]
# `join` doesn't automatically convert ints to strings
','.join(str(x) for x in a)
# 'a,b,c,d,4'
```

Peter Norvig's Spelling Corrector - http://norvig.com/spell-correct.html

One-line Tree in Python (https://gist.github.com/hrldcpr/2012250)
```python
from collections import defaultdict
tree = lambda: defaultdict(tree)

t = tree()
t['EURUSD']['bid'] = 1.1
t['EURUSD']['ask'] = 1.2
```

Largest and smallest elements
```python
import heapq
import random

a = [random.randint(0, 100) for _ in range(100)]
heapq.nsmallest(5, a)
# [3, 3, 5, 6, 8]
heapq.nlargest(5, a)
# [100, 100, 99, 98, 98]
```

k-th largest with min-heap
```python
from heapq import heappop, heappush, nlargest
import random

def select(arr, k):
    """Return the k-th largest element in arr."""
    heap = []
    for x in arr:
        if len(heap) < k or x > heap[0]:
            if len(heap) == k: heappop(heap)
            heappush(heap, x)
    return heap[0]

n = 1000
k = 10
arr = random.sample(range(n), n)
assert select(arr, k) == nlargest(k, arr)[-1] == 990
```

Generators and Coroutines
```python
# Generator
def countdown(n):
    print("Counting down from {}".format(n))
    while n > 0:
        yield n
        n -= 1

counter = countdown(10)
print counter
# <generator object countdown at 0x10dcd6b40>
counter.next()
# Counting down from 10
# 10
print [count for count in counter]
# [9, 8, 7, 6, 5, 4, 3, 2, 1]

# Coroutine
def grep(pattern):
    print("Searching for {}".format(pattern))
    while True:
        line = (yield)
        if pattern in line:
            print(line)

search = grep('coroutine')
print search
# <generator object grep at 0x10dcd6be0>
next(search)
# Searching for coroutine
search.send("I love you")
search.send("Don't you love me?")
search.send("I love coroutines instead!")
# I love coroutines instead!
```

Conway's Game of Life
```python
from itertools import islice

def neighbors(cell):
    x, y = cell
    adj_coord = ((-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1))
    return ((x+dx, y+dy) for dx, dy in adj_coord)

def iterations(board):
    while True:
        new_board = set([])
        candidates = board.union(set(n for cell in board for n in neighbors(cell)))
        for cell in candidates:
            count = sum((n in board) for n in neighbors(cell))
            if count == 3 or (count == 2 and cell in board):
                new_board.add(cell)
        board = new_board
        yield board

if __name__ == "__main__":
    initial_board = {(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    print list(islice(iterations(initial_board), 10))
```

Enum
```
class Color:
    RED, GREEN, BLUE = range(3)

Color.GREEN # 1
```
