---
url: https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html
crawled_at: 2025-11-13T17:23:28.976098
title: https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Download notebook ](https://raw.githubusercontent.com/mitsuba-renderer/mitsuba-tutorials/master/quickstart/drjit_quickstart.ipynb) [ Download data ](https://d38rqfq1h7iukm.cloudfront.net/scenes/tutorials/scenes.zip)
# Dr.Jit quickstart[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Dr.Jit-quickstart "Link to this heading")
## Overview[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Overview "Link to this heading")
This short tutorial recaps the basic functionalities and routines of the Dr.Jit library. You can also find more information on the [Dr.Jit documentation](https://drjit.readthedocs.io).
## Similarity with NumPy[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Similarity-with-NumPy "Link to this heading")
On the Python side, the Dr.Jit _syntax_ is very similar to NumPy. Moreover, as we will see later, both frameworks are interoperable.
Let’s first import both NumPy and Dr.Jit using the alias `np` and `dr` respectively
Copy to clipboard
```
importnumpyasnp
importdrjitasdr

```
Copy to clipboard
Unlike NumPy, Dr.Jit can perform array arithmetic on both CPU and GPU through various template variants which are exposed in top-level packages:
Variant | Description  
---|---  
`drjit.scalar` | Arrays built on top of scalars (float, int, etc.)  
`drjit.llvm` | Arrays built on top of LLVMArray  
`drjit.cuda` | Arrays built on top of CUDAArray  
`drjit.llvm.ad` | Similar to `drjit.llvm` but with automatic differentiation support  
`drjit.cuda.ad` | Similar to `drjit.cuda` but with automatic differentiation support  
These packages all contains various types like: `Bool, Float, Int, UInt, Array2f, Array2i, Matrix2f Matrix3f, ...`
Let’s create some arrays using the `drjit.llvm` variants and play around with the NumPy interoperability:
Copy to clipboard
```
fromdrjit.llvmimport Float, UInt32

# Create some floating-point arrays
a = Float([1.0, 2.0, 3.0, 4.0])
b = Float([4.0, 3.0, 2.0, 1.0])

# Perform simple arithmetic
c = a + 2.0 * b

print(f'c -> ({type(c)}) = {c}')

# Convert to NumPy array
d = np.array(c)

print(f'd -> ({type(d)}) = {d}')

```
Copy to clipboard
```
c -> (<class 'drjit.llvm.Float'>) = [9.0, 8.0, 7.0, 6.0]
d -> (<class 'numpy.ndarray'>) = [9. 8. 7. 6.]

```
Copy to clipboard
## Array construction routines[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Array-construction-routines "Link to this heading")
This section provides an overview of various Dr.Jit routines (and their NumPy correspondence) to construct arrays.
Copy to clipboard
```
# Initialize floating-point array of size 5 with zeros
a = dr.zeros(Float, 5) # np.zeros(5)
print(f'dr.zeros: {a}')

# Initialize floating-point array of size 5 with a constant value
a = dr.full(Float, 0.1, 5) # np.ones(5, 0.4)
print(f'dr.full: {a}')

a = dr.arange(UInt32, 5) # np.arange(5)
print(f'dr.arange: {a}')

# Return evenly spaced numbers over a specified interval
a = dr.linspace(Float, 0.0, 2.0, 5) # np.linspace(0.0, 2.0, 5)
print(f'dr.linespace: {a}')

```
Copy to clipboard
```
dr.zeros: [0.0, 0.0, 0.0, 0.0, 0.0]
dr.full: [0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612]
dr.arange: [0, 1, 2, 3, 4]
dr.linespace: [0.0, 0.5, 1.0, 1.5, 2.0]

```
Copy to clipboard
## Masking[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Masking "Link to this heading")
Writing codes using Dr.Jit often means working with large arrays at once. Therefore it is not possible to use regular `if .. else ..` statements based on concret values, as different elements in the array might branch differently. This is where **masking** comes to the rescue!
A mask (or `Bool`) is an array of boolean values that can be used to disable arithmetic operations on part of an array. It is possible to create such masks with any regular boolean arithmetic (e.g. `>, <, >=, <=`).
Often time, we combine masks with the `dr.select(mask, a, b)` statement which correspond to the ternary statement `mask ? a : b`. This is similar to the `np.where` function in NumPy.
Copy to clipboard
```
x = dr.arange(Float, 5)
m = x > 2.0 # True for all values of a that are greater than 2.0
y = dr.select(m, 4.0, 1.0) # Set the values greater than 2.0 to 4.0 otherwise to 1.0
print(f'x -> ({type(x)}) {x}')
print(f'm -> ({type(m)}) {m}')
print(f'y -> ({type(y)}) {y}')

```
Copy to clipboard
```
x -> (<class 'drjit.llvm.Float'>) [0.0, 1.0, 2.0, 3.0, 4.0]
m -> (<class 'drjit.llvm.Bool'>) [False, False, False, True, True]
y -> (<class 'drjit.llvm.Float'>) [1.0, 1.0, 1.0, 4.0, 4.0]

```
Copy to clipboard
## Basic math arithmetic[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Basic-math-arithmetic "Link to this heading")
All common math operators like `+, -, /, *, *=, +=, %, //, ...` are supported with Dr.Jit arrays.
Similarly to NumPy, Dr.Jit provides all kinds of math arithmetic that can be performed on the entire array in a single call. Here is a non-exaustive list of those math functions: `abs, minimum, maximum, sqrt, pow, sin, cos, tan, atan2, sincos, sec, cot, asin, acos, atan, exp, exp2, log, log2, sinh, cosh, tanh, asinh, acosh, atanh, ...`
Those routines are present in the root drjit package, hence can be used as follow:
Copy to clipboard
```
s, c = dr.sincos(a)
m = dr.minimum(s, c)
print(f'm: {m}')

```
Copy to clipboard
```
m: [0.0, 0.4794255495071411, 0.5403022766113281, 0.07073719799518585, -0.41614681482315063]

```
Copy to clipboard
## Horizontal operations[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#Horizontal-operations "Link to this heading")
Dr.Jit also provides operations that require a pass over the entire array and return a single scalar value. Those operations are expensive as they will trigger a syncronization point, hence it is better to avoid them if possible.
The following snippet of code explores a few of those:
Copy to clipboard
```
a = dr.arange(Float, 5) + 1
print(f'a: {a}')

# Horizontal sum
b = dr.sum(a) # np.sum(a)
print(f'dr.sum(a): {b}')

# Horizontal product
b = dr.prod(a) # np.prod(a)
print(f'dr.prod(a): {b}')

# Mean value over the entire array
b = dr.mean(a) # np.mean(a)
print(f'dr.mean(a): {b}')

m = a > 2
print(f'm: {m}')

# True if all value of the mask array are True
b = dr.all(m) # np.all(m)
print(f'dr.all(m): {b}')

# True if any value of the mask array are True
b = dr.any(m) # np.any(m)
print(f'dr.any(m): {b}')

# True if no value of the mask array are True
b = dr.none(m) # ~np.any(m)
print(f'dr.none(m): {b}')

```
Copy to clipboard
```
a: [1.0, 2.0, 3.0, 4.0, 5.0]
dr.sum(a): [15.0]
dr.prod(a): [120.0]
dr.mean(a): [3.0]
m: [False, False, True, True, True]
dr.all(m): False
dr.any(m): True
dr.none(m): False

```
Copy to clipboard
##  `gather` and `scatter` routines[¶](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html#gather-and-scatter-routines "Link to this heading")
In programming languages like C++ or Python, it is possible to access the i-th element of an array using the `array[i]` syntax. This can both be used to read or write values in an array. Similarly, Dr.Jit provides such read/write functionalities through the `dr.gather` and `dr.scatter` functions. Those are much more powerful than the regular array accessors as the index `i` can be an array itself! In which case the read operation (e.g. `dr.gather`) would return a array as well, not just a single value.
Here is how one should use the `dr.gather` routine to read entries from an Dr.Jit array:
Copy to clipboard
```
source = dr.linspace(Float, 0, 1, 5)
indices = UInt32([1, 2]) # Only read the 2nd and 3rd elements of the source array
result = dr.gather(Float, source, indices)
print(f'source: {source}')
print(f'indices: {indices}')
print(f'result: {result}')

```
Copy to clipboard
```
source: [0.0, 0.25, 0.5, 0.75, 1.0]
indices: [1, 2]
result: [0.25, 0.5]

```
Copy to clipboard
And here is how one can write entries at specific indices into a Dr.Jit array
Copy to clipboard
```
target = dr.zeros(Float, 5)
indices = UInt32([0, 3, 4]) # Write to the first and last two elements of the target array
source = Float([1.0, 2.0, 3.0])
dr.scatter(target, source, indices)
print(f'indices: {indices}')
print(f'source: {source}')
print(f'target: {target}')

```
Copy to clipboard
```
indices: [0, 3, 4]
source: [1.0, 2.0, 3.0]
target: [1.0, 0.0, 0.0, 2.0, 3.0]

```
Copy to clipboard
[Ensure the availability of your data with coverage across AWS, Azure, and GCP on MongoDB Atlas.](https://server.ethicalads.io/proxy/click/9502/019a7c86-dceb-72a2-98cd-e03ae5fb0559/)
[Ads by EthicalAds](https://www.ethicalads.io/advertisers/topics/data-science/?ref=ea-text)
* * *
