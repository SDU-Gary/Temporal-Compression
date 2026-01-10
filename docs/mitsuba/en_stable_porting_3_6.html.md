---
url: https://mitsuba.readthedocs.io/en/stable/porting_3_6.html
crawled_at: 2025-11-13T17:15:38.785789
title: https://mitsuba.readthedocs.io/en/stable/porting_3_6.html
---

Contents Menu Expand Light mode Dark mode Auto light/dark, in light mode Auto light/dark, in dark mode
Hide navigation sidebar
Hide table of contents sidebar
[Skip to content](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#furo-main-content)
Toggle site navigation sidebar
[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/index.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
[ Back to top ](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html)
Toggle Light / Dark / Auto color theme
Toggle table of contents sidebar
# Porting to Mitsuba 3.6.0[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#porting-to-mitsuba-3-6-0 "Link to this heading")
Mitsuba 3.6.0 contains a number of significant changes relative to the prior release, some breaking, that are predominantly driven by the dependence to the new version of [Dr.Jit 1.0.0](https://drjit.readthedocs.io/en/latest/).
This guide is intended to assist users of prior releases of Mitsuba 3 to quickly update their existing codebases to be compatible with Mitsuba 3.6.0 and we further highlight some key changes that are potential pitfalls.
This guide is by no means comprehensive and we direct users to the [Dr.Jit documentation](https://mitsuba.readthedocs.io/en/stable/dr_main) that contains several dedicated sections on the design and core features of Dr.Jit 1.0.0 that Mitsuba users will find invaluable. Historically, the Dr.Jit documentation in past releases has been sparse so it’s recommended that even more advanced users begin here.
## Symbolic control flow[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#symbolic-control-flow "Link to this heading")
[Symbolic loops](https://drjit.readthedocs.io/en/latest/cflow.html#symbolic-mode) in Mitsuba are no longer initialized by constructing a `mitsuba.Loop` instance. Instead, Dr.Jit 1.0.0 introduces the [`drjit.syntax()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.syntax "\(in drjit\)") function decorator that allows users to express symbolic loops as if they were immediately-evaluated Python control flow. An example of a simple loop using `mitsuba.Loop` previously looked like,
```
importmitsubaasmi

var = mi.Float(32)
rng = mi.PCG32(size=102400)

deffoo(var, rng):
  count   = mi.UInt(0)
  loop    = mi.Loop(state=lambda: (var, rng, count))

  while loop(count < 10):
    var     += rng.next_float32()
    count   += 1

  return var, rng

var, rng = foo(var, rng)
var += 1

```
Copy to clipboard
and porting this to use the [`drjit.syntax()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.syntax "\(in drjit\)") decorator is relatively straightforward,
```
importdrjitasdr
importmitsubaasmi

var = mi.Float(32)
rng = mi.PCG32(size=102400)

@dr.syntax
deffoo(var, rng):
  count = mi.UInt(0)

  while count < 10:
    var     += rng.next_float32()
    count   += 1

  return var, rng

var, rng = foo(var, rng)
var += 1

```
Copy to clipboard
Previously when using `mitsuba.Loop`, it was necessary for the user to determine the loop variables required, which was bug-prone. [`drjit.syntax()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.syntax "\(in drjit\)") automates this step and internally reexpresses the loop as a [`drjit.while_loop()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.while_loop "\(in drjit\)") call. While the Dr.Jit API reference details how to directly call [`drjit.while_loop()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.while_loop "\(in drjit\)"), users should prefer using [`drjit.syntax()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.syntax "\(in drjit\)") when porting their existing code.
Prior to Dr.Jit 1.0.0, tracing if-statements containing JIT variables was not possible, and the only alternative was to replace conditionals with masked operations, for example using [`drjit.select()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.select "\(in drjit\)"). As experience has shown, converting conditional code into masked form can be rather tedious and bug-prone.
Therefore, the `dr.syntax()` annotation additionally handles if-statements analogously to while loops, where internally such statements are reexpressed as [`drjit.if_stmt()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.if_stmt "\(in drjit\)") calls. Masked code remains valid but it is often no longer needed.
Warning
While changing existing codebases to leverage symbolic if-statements can improve both readability and performance, it’s important to highlight computational differences relative to [`drjit.select()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.select "\(in drjit\)"). As a contrived example consider
```
x = dr.arange(mi.Float, 5)
y = dr.select(x < 2, 1, 2)

```
Copy to clipboard
which if we were to unwisely express as an if-statement
```
# Don't do this!
@dr.syntax
defbad_code(x : mi.Float):
  out : mi.Float  = mi.Float(0)
  if x < 2:
    out = mi.Float(1)
  else
    out = mi.Float(2)

  return out

x = dr.arange(mi.Float, 5)
y = bad_code()

```
Copy to clipboard
is not only more cumbersome to write but will also give you worse performance relative to the [`drjit.select()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.select "\(in drjit\)") call. This is because now during evaluation, we have to check the condition, perform a jump to either the true or false branch of the if-statement and _then_ step through the branch to perform the output assignment. In contrast, evaluating a [`drjit.select()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.select "\(in drjit\)") call involves no additonal branching.
The real performance benefit of symbolic if-statements are when you have relatively expensive operations that only need to be computed within a given branch, because unlike [`drjit.select()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.select "\(in drjit\)") calls, computations for both the true or false conditions do not have to be evaluated _prior_ to evaluating the if-statement itself. In other words, you can potentially avoid a lot of expensive, branch-specific computations when the condition for evaluating a particular branch is relatively rare.
## Bitmap textures: Half-precision storage by default where possible[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#bitmap-textures-half-precision-storage-by-default-where-possible "Link to this heading")
> Dr.Jit 1.0.0 includes support for half-precision arrays and tensors, and further extends support for FP16 [Dr.Jit textures](https://drjit.readthedocs.io/en/latest/textures.html) that are hardware-accelerated on CUDA backends.
> From Mitsuba 3.6.0 onwards, bitmap textures initialized from data with bit depth 16 or lower will instantiate an underlying half-precision Dr.Jit texture.
> Note
> Using spectral Mitsuba variants is an exception to this default behavior, and the underlying storage of the bitmap texture will remain consistent to the variant as with previous versions of Mitsuba 3. This is because here sampling a texture requires [spectral upsampling](https://rgl.epfl.ch/publications/Jakob2019Spectral) and RGB input data is first converted to their corresponding spectral coefficients.
> There may be cases where this default behavior is undesirable. For instance, if a user is performing an iterative optimization of a given bitmap texture, a potential pitfall is highlighted in the following example
> ```
importmitsubaasmi
importdrjitasdr
mi.set_variant('cuda_ad_rgb')

# Bit depth of my_image.png is less than 16 so storage of texture is FP16
bitmap = mi.load_dict({
    "type" : "bitmap",
    "filename" : "my_image.png"
})

params = mi.traverse(bitmap)

# Want to update the associated tensor but using TensorXf (single-precision)
x = dr.ones(mi.TensorXf, shape=(9,10,3))

# Implicit conversion from TensorXf to TensorXf16
params['data'] = x
params.update()

type(params['data']) # TensorXf16 not TensorXf

```
Copy to clipboard
> The above example is somewhat contrived because in practice, for an optimization, a user would likely initialize their bitmap texture from a tensor and hence the underlying storage precision would be explicitly specified. Regardless, opting out of this default behavior is possible by setting the plugin `format` parameter to `variant`
> ```
importmitsubaasmi
mi.set_variant('cuda_ad_rgb')

# Storage precision is consistent with variant specified (i.e. float)
bitmap = mi.load_dict({
    "type" : "bitmap",
    "filename" : "my_image.png"
    "format" : "variant"
})

params = mi.traverse(bitmap)
type(params['data']) # TensorXf

```
Copy to clipboard
## C++ interface changes[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#c-interface-changes "Link to this heading")
Mitsuba 3.6.0 has also introduced changes that affect C++ developers who have extended Mitsuba 3. As with the Python interface, most of these changes are driven by Dr.Jit 1.0.0 and we again recommend users first begin by reading the [Dr.Jit documentation](https://drjit.readthedocs.io/en/latest/) and in particular the dedicated section on the [Dr.Jit C++ interface](https://drjit.readthedocs.io/en/latest/cpp.html).
### Control flow[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#control-flow "Link to this heading")
Analogous to Dr.Jit’s vectorized control flow changes in Python, in C++ `drjit.Loop` has similarly been removed in Dr.Jit 1.0.0. Here, however there is no equivalent to the Python Dr.Jit function decorator [`drjit.syntax()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.syntax "\(in drjit\)") that automatically tracks which JIT variables are used by the loop. Instead, users are required to call [`drjit.while_loop()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.while_loop "\(in drjit\)") and, as with past releases, manually specify the loop variables
```
Floatx;
Booly;

dr::tie(x,y)=dr::while_loop(dr::make_tuple(x,y),/* initial state */
[](constFloat&x,constBool&y){returny;},/* condition     */
[](Float&x,Bool&y){...});/* body          */

x+=1;

```
Copy to clipboard
Note
Expressing a loop with a high number of tracked variables can be cumbersome to write out. However, Dr.Jit 1.0.0 provides the ability to locally define [custom traversable data types](https://drjit.readthedocs.io/en/latest/cpp.html#custom-types-cpp) that can be leveraged to specify the entire loop state
```
structLoopState{
Floatfoo;
Floatbar;
Floatmore;
Boolactive;
}=ls{x1,x2,x3,active};

dr::tie(ls)=dr::while_loop(dr::make_tuple(ls),/* initial state */
[](constLoopState&ls){returnls.active;},/* condition     */
[](LoopState&ls){...});/* body          */

```
Copy to clipboard
As with the Python interface, the C++ interface similarly exposes support for vectorized conditionals using [`drjit.if_stmt()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.if_stmt "\(in drjit\)") and we direct users to the [Dr.Jit documentation](https://drjit.readthedocs.io/en/latest/cpp.html#vectorized-conditionals) for further details and example usage.
### Removal of `dr::eq`, `dr::neq`[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#removal-of-dr-eq-dr-neq "Link to this heading")
Historically, the Dr.Jit functions `drjit::eq` and `drjit::neq` performed elementwise comparisons on array types
```
Floata,b=...;
Floatres=dr::eq(a,b);

```
Copy to clipboard
while the operators `==` and `!=` would implicitly evaluate and reduce the result
```
boolres=a==b;

```
Copy to clipboard
Dr.Jit 1.0.0 removes `drjit::eq` and `drjit::neq` which are replaced by the overloaded operators `==` and `!=` respectively. Any reductions now have to be explicitly specified by using the [`drjit.all()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.all "\(in drjit\)") or [`drjit.any()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.any "\(in drjit\)") functions for instance
```
boolres=dr::all(a==b);

```
Copy to clipboard
###  `dr::Matrix` ordering now row-major[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#dr-matrix-ordering-now-row-major "Link to this heading")
In Dr.Jit 1.0.0, the internal storage of `dr::Matrix` types has changed from column to row-major ordering. While common matrix operations such as multiplication are unaffected by this change, there is a potential pitfall for existing codebases that read or modify the storage directly, for example
```
dr::Matrix<Float,3>m=...;

// Returned array is now first row, not column!
auto&v=m.entry(0);

```
Copy to clipboard
### Simplified vectorized method getters: `dr::set_attr` removed[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#simplified-vectorized-method-getters-dr-set-attr-removed "Link to this heading")
In past Mitsuba releases, defining custom C++ plugins with [vectorized getters](https://mitsuba.readthedocs.io/en/stable/dr_vcall_get) was bug-prone as developers would be required to additionally remember to call `dr::set_attr` during initialization
```
MyPlugin(constProperties&props):Base(props){
...
m_getter=m_components[0];
dr::set_attr(this,"getter",m_getter);
}

```
Copy to clipboard
which allowed Dr.Jit to perform an optimization during tracing of getters to avoid any actual method calls. Specifically, as getters are read-only and have no side-effects, tracing of such calls can be interpreted as indexing into an array of variables that correspond to the result of each possible instance.
In Dr.Jit 1.0.0, such an optimization remains however developers are no longer required to additionally call `dr::set_attr`.
```
// Registered getter as DRJIT_CALL_GETTER
uint32_tBase::getter()const{returnm_getter;}

MyPlugin(constProperties&props):Base(props){
...
m_getter=m_components[0];
}

```
Copy to clipboard
## Miscellaneous[¶](https://mitsuba.readthedocs.io/en/stable/porting_3_6.html#miscellaneous "Link to this heading")
  * Dr.Jit v1.0.0 raises the minimum supported LLVM version to 11
  * Rename of function `drjit.clamp` to [`drjit.clip()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.clip "\(in drjit\)")
  * Rename of function `drjit.sqr` to `drjit.square()`
  * Rename of function decorator `drjit.wrap_ad` to [`drjit.wrap()`](https://drjit.readthedocs.io/en/latest/reference.html#drjit.wrap "\(in drjit\)")


* * *
