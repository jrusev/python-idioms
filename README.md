Reverse a list
```python
l = [0,1,2,3,4,5]
# Create a new copy with the elements reversed (slicing).
rev = l[::-1]

# Get an iterator that walks the original list in reverse order.
print list(x*2 for x in reversed(l))

# Reverse in place.
l.reverse()
```

Add element(s) to a list.
```python
# Append an item to the end.
a = [1, 2, 3]
a.append([4, 5]) # returns None
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
a = ['a', 'b', 'c', 'b']
a.remove('b')
print a
# ['a', 'c', 'b']

# Remove all occurences of an element in a list.
a = [1, 2, 3, 4, 2]
a = [x for x in a if x != 2]
print a
# [1, 3, 4]

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

Find substring
```python
'hello world'.find('world') # 6
'hello world'.index('world') # 6

'hello world'.find('peace') # -1
'hello world'.index('peace') # ValueError: substring not found
```

List slice assignment
```python
a = [1, 2, 3, 4, 5]
a[2:3] = [0, 0]
print a
# [1, 2, 0, 0, 4, 5]
a[1:1] = [8, 9] # insert
print a
# [1, 8, 9, 2, 0, 0, 4, 5]
a[1:-1] = [] # delete
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
names = ['raymond', 'rachel', 'matthew', 'roger', 'betty', 'melissa', 'judith', 'charlie']

d = defaultdict(list)
for name in names:
    d[len(name)].append(name)

# {5: ['roger', 'betty'], 6: ['rachel', 'judith'], 7: ['raymond', 'matthew', 'melissa', 'charlie']}

# Or just using the standard dict:
d = {}
for name in names:
    key = len(name)
    d[key] = d.get(key, [])
    d[key].append(name)
```

Sort characters in a string by frequency
```python
from collections import Counter
text = 'Mississippi'

dict = Counter(text)
sorted(dict.items(), key=lambda x: -x[1])
# [('i', 4), ('s', 4), ('p', 2), ('M', 1)]

Counter(text).most_common()
# [('i', 4), ('s', 4), ('p', 2), ('M', 1)]

text = 'Mississippi'
sorted((-text.count(w), w) for w in set(text))
# [(-4, 'i'), (-4, 's'), (-2, 'p'), (-1, 'M')]
```

Get all lowercase alphanumeric chars from a string
```python
text = "Hi, John, this is me - Paul. What's up man?"
''.join(ch.lower() for ch in text if ch.isalnum())
# 'hijohnthisismepaulwhatsupman'
```

Remove all spaces from string
```python
text = text.replace(' ', '')
```

Remove all whitespace characters (space, tab, newline, and so on)
```python
text = ''.join(text.split())
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

Iterators for efficient looping
https://docs.python.org/2.7/library/itertools.html

```python
from itertools import combinations
list(''.join(comb) for comb in combinations('ABCD', 3))
# ['ABC', 'ABD', 'ACD', 'BCD']
```

Create a deck of cards
```python
from collections import namedtuple
from itertools import product

Card = namedtuple('Card', 'rank suit')
RANKS = '23456789TJQKA'
SUITS = ('Clubs', 'Diamonds', 'Hearts', 'Spades')
DECK = tuple(Card(*card) for card in product(RANKS, SUITS))
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
    print('Strange, sqrt(-1) did not raise an error!')
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

Print the alphabet
```python
print ''.join(chr(i) for i in range(ord('a'),ord('z')+1))

import string
print string.lowercase
```

Dict Comprehensions
```python
d = {i+1 : chr(ord('A')+i) for i in range(4)}
# {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

d_inverted = {v:k for k, v in d.items()}
# {'A': 1, 'C': 3, 'B': 2, 'D': 4}

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
def fib_gen(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

list(fib_gen(10))
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

[fib(i) for i in range(10)]
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
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

Print a comma separated list of items
```python
a = ['a','b','c','d', 4]
# `join` doesn't automatically convert ints to strings
','.join(str(x) for x in a)
# 'a,b,c,d,4'
```

Largest and smallest elements with [heap](https://docs.python.org/2/library/heapq.html)
```python
from heapq import nsmallest, nlargest
import random

a = [random.randint(0, 100) for _ in range(100)]
nsmallest(5, a)
# [3, 3, 5, 6, 8]
nlargest(5, a)
# [100, 100, 99, 98, 98]
# The latter two functions perform best for smaller values of n. For larger
# values, it is more efficient to use the sorted() function. Also, when n==1,
# it is more efficient to use the built-in min() and max() functions. If
# repeated usage of these functions is required, consider turning the iterable
# into an actual heap
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

Enum
```
class Color:
    RED, GREEN, BLUE = range(3)

Color.GREEN # 1
```

For inspiration, check Peter Norvig's Spelling Corrector - http://norvig.com/spell-correct.html
