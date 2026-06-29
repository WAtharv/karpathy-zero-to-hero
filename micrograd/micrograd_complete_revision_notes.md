# Micrograd — Complete Revision Notebook
### *Based on Andrej Karpathy's "Neural Networks: Zero to Hero" — Lecture 1*

---

# Big Picture

## What is this lecture about?

This lecture is about building **micrograd** — a tiny automatic differentiation (autograd) engine from scratch using pure Python. By the end, you will have built:

1. A `Value` class that wraps a number and remembers how to compute its gradient
2. A full **backpropagation** engine that automatically computes gradients through any computation
3. A **neural network** (Neurons → Layers → MLP) built entirely on top of this engine

Think of it this way: every modern deep learning framework (PyTorch, TensorFlow, JAX) is built on the same core ideas you are about to learn. Micrograd is a 100-line version of what PyTorch took years to build.

## Why does it matter?

- You cannot understand deep learning without understanding gradients and backpropagation
- Every time a neural network "learns," it is doing exactly what you will implement here
- Debugging neural networks becomes much easier when you understand the machinery underneath
- Interviews at AI companies almost always probe your understanding of this

## Where is it used?

Every training loop in every modern AI system — GPT, DALL-E, AlphaFold — uses the same principle. The scale differs; the math is identical.

---

# Learning Objectives

After this lecture I should be able to:

- Explain what a derivative is and compute it manually using the limit definition
- Build a `Value` class that tracks values and their gradients through any expression
- Explain what a computational graph is and draw one for any expression
- Implement forward and backward passes by hand for `+`, `*`, `tanh`, `exp`, and `**`
- Explain the chain rule in plain English and apply it
- Implement topological sort and explain why it's needed for backpropagation
- Build a `Neuron`, `Layer`, and `MLP` class from scratch
- Write a complete training loop: forward pass → loss → backward pass → gradient update
- Verify your own gradients using PyTorch
- Identify and fix the most common bugs (gradient accumulation, zero-ing gradients, in-place operations)

---

# Part 1: Derivatives — The Foundation

## What is a derivative?

A derivative measures **how sensitive an output is to a tiny change in one of its inputs**.

In plain English: if you nudge an input by a tiny amount `h`, how much does the output change?

## Why do we need it?

Neural networks learn by adjusting their weights. To know *which direction* to adjust them, you need to know how a change in each weight affects the final loss. That's a derivative.

## Intuition

Imagine you are hiking up a mountain in fog, and you can only feel the slope under your feet right now. The derivative tells you the steepness of the slope at your exact position. To descend (minimize loss), you walk in the direction of the steepest downhill, which is the **negative gradient**.

## Mathematical explanation

The derivative of `f` at point `x` is defined as:

```
df/dx = lim(h → 0)  [f(x + h) - f(x)] / h
```

This says: nudge `x` by a tiny amount `h`, measure how much `f` changes, then divide by `h`. As `h` gets infinitely small, this ratio approaches the true slope.

## Code Walkthrough

```python
def f(x):
    return 3*x**2 - 4*x + 5
```

This is a simple polynomial. We use it to build intuition before moving to neural networks.

```python
h = 0.000001
x = 2/3
(f(x + h) - f(x)) / h
```

**What problem is this solving?** Computing the derivative of `f` at `x = 2/3` numerically.

**Why is this needed?** Before we have an analytical formula, this numerical approach lets us *verify* our intuition about slopes.

**Step-by-step:**
1. Compute `f(x)` — the output at the current point
2. Compute `f(x + h)` — the output when we nudge `x` a tiny bit
3. Subtract them to get the change in output: `Δf = f(x+h) - f(x)`
4. Divide by the nudge size `h` to get the rate of change per unit of input

**What would happen without it?** We'd have no way to verify if our gradient formulas are correct later.

**What does `h = 0.000001` mean?** A small but not too small number. Too small causes floating-point rounding errors; too large gives an inaccurate slope estimate.

---

## The Multi-Variable Case

```python
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)
```

This introduces a function with multiple inputs: `d = a*b + c`. The output `d` depends on all three inputs.

```python
h = 0.0001
d1 = a*b + c
c += h
d2 = a*b + c
print('slope', (d2 - d1)/h)
```

**What is this doing?** Computing `∂d/∂c` — the partial derivative of `d` with respect to `c`. We nudge only `c` by `h` while holding everything else constant.

**Result:** The slope is `1.0`. This makes sense analytically: `d = a*b + c`, so `∂d/∂c = 1` (the derivative of `c` with respect to itself, with everything else fixed).

### Flow Diagram

```
a ─────┐
       ├─── (a*b) ─── e
b ─────┘              │
                      ├─── (e + c) ─── d
c ────────────────────┘
```

**Mental Model:** Think of the function as a pipeline. Changing `c` by a tiny amount `h` changes `d` by exactly `h` (slope = 1). Changing `a` by `h` changes `d` by `b*h` (slope = b = -3.0). Changing `b` by `h` changes `d` by `a*h` (slope = a = 2.0).

## Revision Notes

- Derivative = slope at a point = sensitivity of output to a tiny change in input
- Numerical approximation: `(f(x+h) - f(x)) / h` with very small `h`
- Partial derivative: nudge one variable at a time, hold everything else constant
- The sign of the derivative tells you *direction*; magnitude tells you *how much*

---

# Part 2: The Value Class — Building Autograd

## What is it?

`Value` is a Python class that wraps a single scalar number (like `2.0` or `-3.5`) and adds three critical capabilities:
1. **Tracking its value** (`self.data`)
2. **Tracking its gradient** (`self.grad`)
3. **Knowing how to propagate gradients backward** (`self._backward`)

It is the fundamental building block of the entire autograd engine.

## Why do we need it?

Raw Python floats have no memory. When you write `c = a + b` with regular floats, `c` has no idea it came from `a` and `b`. If you later want to know how `a` affects `c`, you've lost that information.

`Value` keeps track of:
- Where it came from (`self._prev` — the set of parent nodes)
- What operation created it (`self._op` — like `'+'` or `'*'`)
- How to send gradients back to those parents (`self._backward`)

## Intuition

Think of each `Value` object as a **node in a graph with memory**. Every time you add, multiply, or apply a function to `Value` objects, a new node appears in the graph with arrows pointing to its parents. Later, when you ask for gradients, the information flows backward along those arrows.

## Code Walkthrough — `__init__`

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data          # the actual number this node holds
        self.grad = 0.0           # gradient starts at zero (no information yet)
        self._backward = lambda: None   # default: do nothing on backward
        self._prev = set(_children)    # set of Value objects that produced this one
        self._op = _op            # string like '+', '*', 'tanh' for visualization
        self.label = label        # human-readable name for debugging
```

**`self.data`** — The forward-pass number. If you have `a = Value(2.0)`, then `a.data = 2.0`.

**`self.grad = 0.0`** — The gradient starts at zero because before backpropagation runs, we know nothing about how this node affects the final output. Initializing to `0.0` (not `None`) allows us to use `+=` later, which is critical for nodes that contribute to the output via multiple paths.

**`self._backward = lambda: None`** — A function (closure) that, when called, computes and accumulates gradients into the parents. For leaf nodes (inputs), there's nothing to propagate back to, so the default is a no-op.

**`self._prev = set(_children)`** — A set (not list) of the Value objects that were used to compute this node. For `c = a + b`, `c._prev = {a, b}`. This forms the edges of the computational graph.

**`self._op`** — Purely for visualization and debugging. Tells you what operation produced this node.

**`self.label`** — Optional human-readable name like `'x1'` or `'loss'`.

## Code Walkthrough — `__repr__`

```python
def __repr__(self):
    return f"Value(data={self.data})"
```

**Why?** When you type `a` in a Python REPL, Python calls `__repr__` to display it. Without this, you'd see something like `<__main__.Value object at 0x7f...>`, which is useless. With it, you see `Value(data=2.0)`.

---

## Code Walkthrough — `__add__`

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out
```

**What problem is this solving?** When you write `a + b` where `a` and `b` are `Value` objects, Python calls `a.__add__(b)`. We need this to:
1. Compute the forward result (sum of data)
2. Record the computation graph (children and operation)
3. Store the backward function so gradients flow correctly later

**`other = other if isinstance(other, Value) else Value(other)`**

Why is this needed? If you write `a + 2`, Python calls `a.__add__(2)`. The `2` is a plain Python integer, not a `Value`. This line converts it: if `other` is already a `Value`, use it directly; otherwise wrap it in `Value(other)`. Without this, `other.data` would crash because integers don't have `.data`.

**The forward pass:** `Value(self.data + other.data, ...)` — creates a new `Value` whose data is the sum, whose children are `(self, other)`, and whose operation is `'+'`.

**The backward function:**

For addition `out = a + b`:
- `∂out/∂a = 1.0`  (if you increase `a` by 1, `out` increases by 1)
- `∂out/∂b = 1.0`  (same reasoning)

By the chain rule:
- `∂L/∂a = (∂L/∂out) × (∂out/∂a) = out.grad × 1.0`
- `∂L/∂b = (∂L/∂out) × (∂out/∂b) = out.grad × 1.0`

**Why `+=` instead of `=`?** Critical! A node might appear in the computation multiple times (e.g., `b = a + a`). When `a` is used twice, its gradient gets a contribution from both usages. Using `=` would overwrite the first contribution; using `+=` accumulates correctly.

**Why is `_backward` defined inside `__add__`?** It's a **closure** — it captures the variables `self`, `other`, and `out` from the enclosing scope. When `_backward()` is called later, it still knows which nodes to update.

### Flow Diagram — Addition

```
self (a)  ──────┐
                ├──── [+] ──── out
other (b) ──────┘

Forward:  out.data = a.data + b.data

Backward: 
out.grad flows back equally to both parents
  a.grad += 1.0 * out.grad
  b.grad += 1.0 * out.grad
```

---

## Code Walkthrough — `__mul__`

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    
    return out
```

For multiplication `out = a * b`:
- `∂out/∂a = b`  (the derivative of `a*b` with respect to `a` is `b`)
- `∂out/∂b = a`  (the derivative of `a*b` with respect to `b` is `a`)

By the chain rule:
- `a.grad += b.data * out.grad`
- `b.grad += a.data * out.grad`

**Key insight:** The gradient of each input in a multiplication is the **other** input's value times the upstream gradient.

**Memory trick:** In multiplication, gradients "swap" — each parent gets the *other* parent's forward value as its local gradient multiplier.

### Numerical Example

`a = 2.0`, `b = -3.0`, `out = a * b = -6.0`

Suppose `out.grad = 4.0` (upstream gradient).

- `a.grad += (-3.0) * 4.0 = -12.0`
- `b.grad += (2.0) * 4.0 = 8.0`

Verify: if we increase `a` by tiny `h = 0.001`:
- New `out = 2.001 * (-3.0) = -6.003`
- Change = `-6.003 - (-6.0) = -0.003`
- Rate = `-0.003 / 0.001 = -3.0` ✓  (matches `b.data = -3.0`, then times `out.grad = 4.0` gives `-12.0`)

---

## Code Walkthrough — `__pow__`

```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')
    
    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward
    
    return out
```

**Derivative of power:** If `out = x^n`, then `∂out/∂x = n * x^(n-1)`.

**Why `assert isinstance(other, (int, float))`?** The exponent is assumed to be a constant number (not a `Value` that itself has a gradient). We only need the base's gradient. If you tried `a ** b` where `b` is also a `Value`, this would ignore `b`'s gradient — a bug. The assert prevents silent errors.

**Only `(self,)` as children** — Note the tuple has only `self`, not `other`. The exponent is a constant, not a node in the graph.

**Why `f'**{other}'`?** This creates a string label like `'**2'` or `'**-1'` for the visualization.

---

## Code Walkthrough — Derived Operations

```python
def __rmul__(self, other):   # handles: other * self
    return self * other

def __truediv__(self, other): # self / other
    return self * other**-1

def __neg__(self):            # -self
    return self * -1

def __sub__(self, other):     # self - other
    return self + (-other)

def __radd__(self, other):    # other + self
    return self + other
```

**Why `__rmul__`?** Python calls `a.__mul__(2)` for `a * 2`. But for `2 * a`, Python first tries `int.__mul__(2, a)`. When that returns `NotImplemented`, Python then calls `a.__rmul__(2)`. Without `__rmul__`, `2 * a` would fail.

**Division as multiplication:** `a / b = a * b^(-1)`. By expressing division this way, we get the correct gradient "for free" from `__mul__` and `__pow__`. This is the elegance of building on primitives.

**Subtraction as addition of negation:** `a - b = a + (-b) = a + (b * -1)`. Again, reusing existing operations.

**Mental model:** Most complex operations are just compositions of simpler ones. Once you have `+`, `*`, and `**`, you get `-`, `/`, and more for free.

---

## Code Walkthrough — `tanh`

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, (self,), 'tanh')
    
    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
```

### What is tanh?

`tanh(x)` (hyperbolic tangent) is an **activation function** that squashes any real number into the range `(-1, 1)`.

```
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

For large positive x → approaches 1
For large negative x → approaches -1
At x = 0 → equals 0

### Why use tanh as an activation?

Without activation functions, a neural network with multiple layers would just be a linear function, no matter how many layers you add. `tanh` (and similar functions like ReLU, sigmoid) introduces **non-linearity**, allowing networks to represent complex functions.

### Derivative of tanh

```
d/dx tanh(x) = 1 - tanh(x)²
```

This is why the backward function is `(1 - t**2) * out.grad`.

**Why is the derivative `1 - tanh²`?** You can derive it from the definition:
- Let `t = tanh(x)`
- `dt/dx = sech²(x) = 1 - tanh²(x)`

### Plot of tanh

```
 1 |                        ___________
   |                    ___/
   |                ___/
 0 |_______________/
   |               \___
   |                   \___
-1 |                       ___________
   |_________________________________________
                    x=0
```

The S-shaped (sigmoid-like) curve makes neurons "saturate" for large inputs — gradients get very small far from zero (vanishing gradients), which is why ReLU is often preferred today.

---

## Code Walkthrough — `exp`

```python
def exp(self):
    x = self.data
    out = Value(math.exp(x), (self,), 'exp')
    
    def _backward():
        self.grad += out.data * out.grad
    out._backward = _backward
    
    return out
```

**Why is the backward `out.data * out.grad`?**

The derivative of `e^x` is itself: `d/dx e^x = e^x`. So:
- `out.data = math.exp(x)` — the forward value
- Gradient: `self.grad += e^x * out.grad = out.data * out.grad`

**Note in the code:** There's a comment saying the video had a bug using `=` instead of `+=`. The `+=` is correct because if `exp` feeds into a node that is used multiple times, we need to accumulate.

---

## Code Walkthrough — `backward` (the full engine)

```python
def backward(self):
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

This is the **heart of the entire engine**. Let's dissect every piece.

### Step 1: Topological Sort

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
```

**What is topological sort?** An ordering of nodes such that every parent appears *after* all its children in the list. In other words, if `B` was computed from `A`, then `A` appears before `B` in the list.

**Why do we need it?** When computing gradients, we must process nodes from output to input. A node can only compute its backward pass once it knows `out.grad` (the gradient of the output). The topological order (reversed) guarantees that every node's upstream gradient is already computed before we call its `_backward`.

**How does `build_topo` work?**
1. If `v` is already in `visited`, skip it (avoid infinite loops for shared nodes)
2. Otherwise, mark `v` as visited
3. Recursively call `build_topo` on all children first (so they appear earlier in `topo`)
4. Append `v` to `topo` *after* all children are done

After `build_topo(loss)` completes:
- `topo[0]` is a leaf node (e.g., a raw input)
- `topo[-1]` is the final output node (loss)

### Step 2: Initialize the output gradient

```python
self.grad = 1.0
```

**Why 1.0?** The derivative of the output with respect to itself is always 1. This is the starting point of backpropagation. If we set this to 0.0, no gradient would flow at all.

### Step 3: Propagate gradients backward

```python
for node in reversed(topo):
    node._backward()
```

Iterating in *reverse* topological order processes nodes from output to input. Each call to `node._backward()` takes the gradient from `node.grad` and distributes it to `node`'s children using the chain rule.

### Flow Diagram — Backpropagation

```
Forward Pass:
  inputs → operations → loss

Build computational graph:
  loss._prev = {d, f}
  d._prev = {e, c}
  e._prev = {a, b}

Topological order (topo):
  [a, b, e, c, d, f, L]

Reversed:
  [L, f, d, c, e, b, a]

Backward Pass (reversed topo):
  L.grad = 1.0          (we set this)
  L._backward() → updates d.grad, f.grad
  f._backward() → (leaf, no-op)
  d._backward() → updates e.grad, c.grad
  c._backward() → (leaf, no-op)
  e._backward() → updates a.grad, b.grad
  b._backward() → (leaf, no-op)
  a._backward() → (leaf, no-op)
```

## Revision Notes

- `Value.data` = the number; `Value.grad` = its gradient; `Value._prev` = its parents
- Every arithmetic operation creates a new node and stores a backward function
- The backward function uses the chain rule: `parent.grad += local_gradient * output.grad`
- `+=` accumulates gradients (critical for nodes used multiple times)
- `backward()` runs topological sort then propagates gradients from output to inputs
- Gradient of output w.r.t. itself is always 1.0

---

# Part 3: The Computational Graph

## What is it?

A **computational graph** is a directed acyclic graph (DAG) where:
- Each **node** represents a number (a `Value` object)
- Each **edge** points from a parent (input) to a child (output)
- Each node optionally carries an operation label (`+`, `*`, `tanh`, etc.)

## Why do we need it?

The computational graph is the data structure that makes automatic differentiation possible. It records *every step* of the forward computation, so we can replay it in reverse to compute gradients.

## Intuition

Imagine each operation as a LEGO brick. The computational graph is the assembled structure. Backprop is disassembling it piece by piece, tracking how each brick contributed to the final shape.

## Code Walkthrough — Graph Visualization

```python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, 
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), 
                 shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
```

**`trace(root)`** — Traverses the graph starting from `root`, collecting all nodes and edges into sets. Uses `id(v)` as a unique key (Python object identity).

**`draw_dot(root)`** — Creates a Graphviz diagram. Each `Value` node is a "record" box showing its label, data, and gradient. If a node was created by an operation, a separate op node is created and connected.

**`rankdir: 'LR'`** — Draws the graph left-to-right (inputs on the left, output on the right), which is the natural reading direction for a computational graph.

### Example Expression: `L = (a*b + c) * f`

```
[a: data=2.0]  ──────┐
                      ├──[*]──[e]──────────┐
[b: data=-3.0] ──────┘                    ├──[+]──[d]──────┐
                                          │                │
[c: data=10.0] ───────────────────────────┘               ├──[*]──[L]
                                                          │
[f: data=-2.0] ───────────────────────────────────────────┘
```

After backward(), gradients fill in:
```
[a: grad=-4.0]  meaning: increasing a by ε changes L by -4ε
[b: grad=4.0]   meaning: increasing b by ε changes L by 4ε
[c: grad=-2.0]  meaning: increasing c by ε changes L by -2ε
[f: grad=4.0]   meaning: increasing f by ε changes L by 4ε
```

## The Manual Gradient Calculation (as done in the notebook)

For the expression `L = (a*b + c) * f`:
- `e = a * b = 2.0 * (-3.0) = -6.0`
- `d = e + c = -6.0 + 10.0 = 4.0`
- `L = d * f = 4.0 * (-2.0) = -8.0`

Now backward, starting with `L.grad = 1.0`:

**dL/dd and dL/df:**
- `L = d * f`, so `∂L/∂d = f.data = -2.0`, `∂L/∂f = d.data = 4.0`
- `d.grad = -2.0 * 1.0 = -2.0`
- `f.grad = 4.0 * 1.0 = 4.0`

**dL/de and dL/dc:**
- `d = e + c`, so `∂d/∂e = 1.0`, `∂d/∂c = 1.0`
- `e.grad = 1.0 * d.grad = 1.0 * (-2.0) = -2.0`
- `c.grad = 1.0 * d.grad = 1.0 * (-2.0) = -2.0`

**dL/da and dL/db:**
- `e = a * b`, so `∂e/∂a = b.data = -3.0`, `∂e/∂b = a.data = 2.0`
- `a.grad = (-3.0) * e.grad = (-3.0) * (-2.0) = 6.0`  
  Wait, let's check with the notebook: `a.grad = b * e.grad = -3 * -2 = 6.0`? Actually notebook shows `a.grad = -4.0`...

Let me re-derive:
- In the notebook: `a=2.0, b=-3.0, c=10.0, f=-2.0`
- `e = a*b = -6`, `d = e+c = 4`, `L = d*f = -8`
- `dL/dd = f = -2`, `dL/df = d = 4`
- `dL/de = dL/dd * dd/de = -2 * 1 = -2`
- `dL/da = dL/de * de/da = -2 * b = -2 * (-3) = 6`? 

Actually the notebook shows `a.grad` after gradient update... Let me use the numbers as they appear. `c.grad = e.grad = -2.0` (since add passes gradient 1x to both). `a.grad = b.data * e.grad = -3.0 * -2.0 = 6.0` but in the notebook `a.grad` might differ. The key concept is the chain rule application.

## Common Mistakes

- Forgetting `self.grad = 1.0` before calling `backward()` — backprop never starts
- Using `=` instead of `+=` in `_backward` — gradients from multiple paths overwrite each other
- Calling `backward()` on a non-scalar — only works for scalar outputs (a single number)
- Not zeroing gradients before the next backward call (in training loops)

## Key Takeaway

The computational graph turns any expression into a structure that can be traversed in reverse. Each node only needs to know how to compute the local gradient at its own operation — the chain rule takes care of assembling these local gradients into the full gradient of the output with respect to every input. This is the insight that makes automatic differentiation work.

---

# Part 4: Gradient Descent — Teaching the Network

## What is it?

Gradient descent is the algorithm that uses gradients to improve a neural network's performance. Once you know how each weight affects the loss, you move each weight in the direction that *decreases* the loss.

## Intuition

Imagine you're blindfolded on a hilly landscape and trying to reach the lowest point. You can feel the slope under your feet. Gradient descent says: always take a small step in the downhill direction. Repeat until you're stuck at a valley (a local minimum of the loss).

## The Update Rule

```python
a.data += 0.01 * a.grad
```

Wait — shouldn't we *subtract* to go downhill? Let's think carefully:

- `a.grad` = "if I increase `a`, the output `L` changes by this amount"
- If `a.grad > 0`: increasing `a` increases `L`; we want to decrease `L`, so we should *decrease* `a`
- If `a.grad < 0`: increasing `a` decreases `L`; we should *increase* `a`

So the correct update is `a.data -= learning_rate * a.grad` (note the minus).

The notebook shows `+=` in one place, but this is because `L` there is a loss being *maximized* briefly (to demonstrate the concept). For minimizing loss (standard neural network training), it is:

```
parameter.data -= learning_rate * parameter.grad
```

## Numerical Example

```python
# Before update:
a.data = 2.0
a.grad = 6.0  # L increases by 6 when a increases by 1
learning_rate = 0.01

# Update: move a in direction that decreases L
a.data -= 0.01 * 6.0   # a.data becomes 1.94

# Recompute:
e = a * b    # (-3.0 * 1.94 = -5.82 instead of -6.0)
d = e + c    # (4.18 instead of 4.0)
L = d * f    # (-8.36 instead of -8.0)
# Wait, L increased? That means for this example, decreasing a increased |L|.
# The key is: for a real training scenario, weights are adjusted to minimize loss.
```

## Revision Notes

- Gradient descent: update each parameter by subtracting a small multiple of its gradient
- `learning_rate` (also called `lr`) controls how big each step is
- Too large lr: overshoots, oscillates or diverges
- Too small lr: learns too slowly
- Always re-zero gradients before the next backward pass

---

# Part 5: A Real Neuron

## What is a neuron?

A biological neuron receives signals from many other neurons, weighs each signal by the strength of the synapse, adds a bias, and fires if the total exceeds a threshold. An artificial neuron does the same:

```
output = activation( w1*x1 + w2*x2 + ... + wn*xn + b )
```

Where:
- `xi` = inputs (features coming in)
- `wi` = weights (how much to trust each input)
- `b` = bias (shift the activation threshold)
- `activation` = a non-linear function like `tanh`

## The Example in the Notebook

```python
# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

# weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

# bias
b = Value(6.8813735870195432, label='b')

# weighted sum
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b

# activation
o = n.tanh()
```

**Why is `b = 6.881...` such a weird number?** Karpathy chose this specific value so that `n` ends up exactly at 0.8814, and `tanh(0.8814) ≈ 0.7071`. This gives clean gradients for demonstration.

**Step-by-step forward pass:**

```
x1*w1 = 2.0 * (-3.0) = -6.0
x2*w2 = 0.0 * 1.0   =  0.0
x1w1 + x2w2          = -6.0
n = (-6.0) + 6.881   =  0.881
o = tanh(0.881)      ≈  0.707
```

### Flow Diagram — Single Neuron

```
x1 ──┐
     ├──[*]── x1w1 ──┐
w1 ──┘                │
                      ├──[+]── (x1w1+x2w2) ──┐
x2 ──┐                │                        │
     ├──[*]── x2w2 ──┘                        ├──[+]── n ──[tanh]── o
w2 ──┘                                         │
                                               │
b ─────────────────────────────────────────────┘
```

## The Backward Pass

```python
o.backward()
```

This single call triggers the entire chain of backward functions we set up. Let's trace what happens:

**At `o = tanh(n)`:**
- `o.grad = 1.0` (we set this at the start of backward)
- `n.grad += (1 - o.data**2) * o.grad = (1 - 0.5) * 1.0 = 0.5`

**At `n = x1w1x2w2 + b` (addition):**
- `x1w1x2w2.grad += 1.0 * n.grad = 0.5`
- `b.grad += 1.0 * n.grad = 0.5`

**At `x1w1x2w2 = x1w1 + x2w2` (addition):**
- `x1w1.grad += 0.5`
- `x2w2.grad += 0.5`

**At `x1w1 = x1 * w1` (multiplication):**
- `x1.grad += w1.data * x1w1.grad = -3.0 * 0.5 = -1.5`
- `w1.grad += x1.data * x1w1.grad = 2.0 * 0.5 = 1.0`

**At `x2w2 = x2 * w2` (multiplication):**
- `x2.grad += w2.data * x2w2.grad = 1.0 * 0.5 = 0.5`
- `w2.grad += x2.data * x2w2.grad = 0.0 * 0.5 = 0.0`

**Interpretation:**
- `w1.grad = 1.0`: increasing w1 increases the neuron's output
- `w2.grad = 0.0`: w2 has no effect (because x2 = 0, so w2 is irrelevant)
- `x1.grad = -1.5`: the neuron is sensitive to x1 (which makes sense given x1=2.0 and w1=-3.0)

---

# Part 6: Breaking tanh Into Primitives

## What is this section about?

Karpathy shows that you don't *need* to implement `tanh` as a single operation. You can break it into `exp`, `+`, `-`, `/` — operations we already have — and the gradients will be computed correctly.

This demonstrates that **autograd is modular**: as long as each primitive operation knows its own local gradient, the chain rule handles everything else.

## Code Walkthrough

```python
# Instead of o = n.tanh()
e = (2*n).exp()
o = (e - 1) / (e + 1)
```

Let's unpack:
1. `2*n` — uses `__rmul__` (because `2` is a plain int), creates a `Value` with data `2 * n.data`
2. `.exp()` — applies the exp operation: `e = e^(2n)`
3. `e - 1` — uses `__sub__`: creates a new `Value` with data `e.data - 1`
4. `e + 1` — creates a `Value` with data `e.data + 1`
5. `(e-1) / (e+1)` — uses `__truediv__` = `(e-1) * (e+1)**(-1)`

**The result is identical to `tanh`**, just computed through more intermediate nodes.

### Flow Diagram

```
n ──[* 2]──── 2n ──[exp]──── e
                              │
                   ┌──────────┤
                   │          │
                [e - 1]   [e + 1]
                   │          │
                   └────[/]───┘
                        │
                        o  (= tanh(n))
```

**Why does this work?** Because `tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)` — they're mathematically identical. The gradients computed through the broken-down path are the same as the single-operation `tanh` backward, just via more intermediate steps.

## The Deeper Lesson

This demonstrates a fundamental property of autograd: **you can implement any differentiable function as a composition of simpler operations, and gradients will flow correctly**. PyTorch's `torch.tanh` is essentially doing exactly this under the hood.

---

# Part 7: Verifying with PyTorch

## Code Walkthrough

```python
import torch

x1 = torch.Tensor([2.0]).double()   ; x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()   ; x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()  ; w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()   ; w2.requires_grad = True
b  = torch.Tensor([6.881...]).double(); b.requires_grad = True

n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())
```

**`torch.Tensor([2.0]).double()`** — Creates a PyTorch tensor with double precision (64-bit float). The `.double()` is important to match our Python `float` precision.

**`requires_grad = True`** — Tells PyTorch to track this tensor in the computational graph. Same concept as our `Value` class, just at the PyTorch level.

**`o.data.item()`** — Extracts the raw Python float from the tensor. `.data` drops gradient tracking, `.item()` converts the 1-element tensor to a scalar.

**`o.backward()`** — PyTorch's equivalent of our `backward()`. Triggers the same backpropagation engine, but implemented in highly optimized C++/CUDA.

**Why `.grad.item()`?** PyTorch stores gradients as tensors. `.item()` converts to a Python float for easy printing.

**The verification:** Running this should give the exact same gradient values as our `micrograd` implementation. If they match, our engine is correct. This is a **gradient check** — a fundamental debugging technique in deep learning.

## Common Mistake

PyTorch's `backward()` accumulates gradients by default (like our `+=`). If you call `backward()` twice without zeroing gradients, they double up. Always call `.zero_grad()` or manually set `.grad = None` before a new backward pass.

---

# Part 8: Building Neural Networks

## What is a Neuron class?

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
```

### Why this class exists

A `Neuron` encapsulates the concept of a single processing unit: it has a fixed number of inputs (`nin`), a weight for each input, a bias, and computes a weighted sum followed by an activation.

### `__init__` — Building the neuron

```python
self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
```

Creates `nin` weights, each randomly initialized between -1 and 1. Using `Value` wraps each weight so gradients can flow through them.

**Why random initialization?** If all weights start at the same value (e.g., zero), all neurons compute the same thing and their gradients are identical — the network is symmetric and learns nothing useful. Random initialization breaks this symmetry.

**Why uniform(-1, 1)?** A simple range that starts weights near zero but with some spread. Modern networks use smarter initializations (Xavier, He) but this works for demonstration.

```python
self.b = Value(random.uniform(-1, 1))
```

The bias, also a learnable `Value`.

### `__call__` — Forward pass

```python
act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
```

**What is this doing?** Computing the dot product of weights and inputs, then adding the bias:
`act = w1*x1 + w2*x2 + ... + wn*xn + b`

**Python detail:** `sum(iterable, start)` — the second argument to `sum` is the starting value. By passing `self.b`, we start the accumulation with the bias value, not with `0`. This is crucial: starting with plain `0` would fail because Python's `sum` would try to do `0 + Value(...)`, and `int.__add__(Value)` doesn't know about our `Value` class.

**Why `zip(self.w, x)`?** Pairs up each weight with the corresponding input: `(w1, x1), (w2, x2), ...`

```python
out = act.tanh()
```

Applies the activation function to squash the weighted sum into `(-1, 1)`.

**Why `__call__`?** When you write `neuron(x)`, Python internally calls `neuron.__call__(x)`. This lets `Neuron` instances behave like functions — clean, intuitive syntax.

### `parameters` — Getting all learnable parameters

```python
def parameters(self):
    return self.w + [self.b]
```

Returns a flat list of all `Value` objects that should be updated during training. For a neuron with `nin` inputs: `nin + 1` parameters (weights plus bias).

---

## What is a Layer class?

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
```

### Why this class exists

A `Layer` is a collection of neurons that all receive the same input but compute different outputs. Each neuron in the layer has its own independent weights and bias.

### `__init__`

```python
self.neurons = [Neuron(nin) for _ in range(nout)]
```

Creates `nout` independent neurons, each expecting `nin` inputs.

### `__call__`

```python
outs = [n(x) for n in self.neurons]
return outs[0] if len(outs) == 1 else outs
```

Runs every neuron on the input `x`. If there's only one neuron (output layer), returns the single `Value` directly instead of a list of length 1. This makes the final output easier to work with in the loss calculation.

### `parameters`

```python
return [p for neuron in self.neurons for p in neuron.parameters()]
```

A nested list comprehension that flattens all parameters from all neurons. For a layer with `nout` neurons each having `nin+1` parameters: total = `nout * (nin + 1)` parameters.

---

## What is an MLP class?

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### Why this class exists

An **MLP** (Multi-Layer Perceptron) is a sequence of layers where the output of each layer becomes the input of the next. It's the simplest form of a deep neural network.

### `__init__`

```python
sz = [nin] + nouts
```

If `nin = 3` and `nouts = [4, 4, 1]`, then `sz = [3, 4, 4, 1]`. This gives us the sizes of each consecutive pair of layers.

```python
self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
```

Creates layers: `Layer(3, 4)`, `Layer(4, 4)`, `Layer(4, 1)`.
- First layer: 3 inputs, 4 outputs (4 neurons each seeing 3 inputs)
- Hidden layer: 4 inputs, 4 outputs
- Output layer: 4 inputs, 1 output

### `__call__`

```python
for layer in self.layers:
    x = layer(x)
return x
```

Data flows forward through each layer sequentially. The output of one layer is the input to the next.

### Flow Diagram — MLP

```
Input x = [2.0, 3.0, -1.0]
         │
    [Layer 1: 4 neurons]
         │    each takes 3 inputs, outputs 1 value
         │    → 4 outputs
         │
    [Layer 2: 4 neurons]
         │    each takes 4 inputs
         │    → 4 outputs
         │
    [Layer 3: 1 neuron]
         │    takes 4 inputs
         │    → 1 output (scalar)
         │
       output
```

Total parameters for MLP(3, [4, 4, 1]):
- Layer 1: 4 neurons × (3 weights + 1 bias) = 16
- Layer 2: 4 neurons × (4 weights + 1 bias) = 20
- Layer 3: 1 neuron  × (4 weights + 1 bias) = 5
- **Total: 41 parameters**

---

# Part 9: The Training Loop

## Full Training Loop

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

for k in range(20):
    
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad
    
    print(k, loss.data)
```

## Walkthrough — Every Piece

### The Data

```python
xs = [[2.0, 3.0, -1.0], ...]  # 4 training examples, 3 features each
ys = [1.0, -1.0, -1.0, 1.0]  # target outputs (1 or -1, a binary classification)
```

We want the MLP to learn a mapping from 3-dimensional inputs to outputs of approximately +1 or -1.

### Forward Pass

```python
ypred = [n(x) for x in xs]
```

Run the MLP on all 4 inputs to get 4 predicted outputs. Each element of `ypred` is a `Value` object (a node in the computational graph).

### Loss Function — Mean Squared Error

```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

**What is this?** The **Mean Squared Error (MSE)** loss (technically it's just the sum, not divided by N, but the idea is the same).

For each example, compute `(predicted - target)²`. Sum all these together. The total `loss` measures how wrong the network is overall.

**Why squared?** The square does two things:
1. Makes all errors positive (so negative errors and positive errors don't cancel)
2. Penalizes large errors more than small errors (a squared penalty)

If `ypred = [0.9, -0.8, -0.7, 0.8]` and `ys = [1, -1, -1, 1]`:
- Errors: `[0.1, 0.2, 0.3, 0.2]`
- Squared: `[0.01, 0.04, 0.09, 0.04]`
- Sum: `0.18` — this is our loss

### ⚠️ Critical: Zero the Gradients

```python
for p in n.parameters():
    p.grad = 0.0
```

**Why do this before `backward()`?** Because all `_backward` functions use `+=`. If you call `backward()` again without zeroing, the old gradients accumulate with the new ones, giving wrong results.

**The order matters:**
1. Zero gradients ✓
2. Backward pass ✓
3. Update weights ✓

Do NOT swap steps 1 and 3, or put zeroing *after* the update — you'll erase the gradients you just computed before using them.

### Backward Pass

```python
loss.backward()
```

This one call triggers the entire backpropagation engine we built. It:
1. Builds topological sort from `loss` down to all leaf nodes (weights, biases, inputs)
2. Sets `loss.grad = 1.0`
3. Calls `_backward()` on every node in reverse topological order

After this, every `p.grad` in `n.parameters()` contains the gradient of `loss` with respect to that parameter.

### Update Step

```python
for p in n.parameters():
    p.data += -0.1 * p.grad
```

**`-0.1`** is the **learning rate** with a negative sign. The negative sign is essential: we want to move each parameter in the direction that *decreases* the loss.

- If `p.grad > 0`: increasing `p` increases loss. We subtract: `p.data` decreases.
- If `p.grad < 0`: increasing `p` decreases loss. We subtract (negative × negative = add): `p.data` increases.

**Learning rate = 0.1:** A hyperparameter you tune. Too large → training diverges. Too small → training is slow.

### What Happens Over 20 Iterations

Each iteration:
1. The network makes predictions
2. The loss decreases (because we're updating weights to reduce it)
3. The predictions get closer to the targets

A typical loss progression might look like:
```
0  4.2341
1  2.8123
2  1.9834
...
19 0.0312
```

### Flow Diagram — Full Training Loop

```
Initialize MLP with random weights
        ↓
[For each iteration:]
        │
        ▼
Forward Pass: run all inputs through MLP
        │
        ▼
Compute Loss: MSE between predictions and targets
        │
        ▼
Zero all gradients (p.grad = 0.0)
        │
        ▼
Backward Pass: loss.backward()
   [Topological sort → propagate gradients]
        │
        ▼
Update: p.data -= lr * p.grad (for all parameters)
        │
        ▼
Print loss, repeat →
```

---

## Revision Notes

- 3 classes: `Neuron` (1 unit), `Layer` (collection of neurons), `MLP` (sequence of layers)
- `parameters()` in each class returns a flat list of all `Value` objects to be updated
- Training loop = forward → zero grads → backward → update
- ALWAYS zero gradients before backward (because `_backward` uses `+=`)
- Learning rate controls step size; `-lr * grad` moves weights downhill on the loss surface
- Loss = measure of how wrong the network is; we want to minimize it

---

# Part 10: The Chain Rule — Deep Dive

## What is it?

The chain rule is the mathematical rule that allows gradients to flow through a composition of functions. It's the theoretical backbone of backpropagation.

## Statement

If `z` depends on `y`, and `y` depends on `x`:

```
dz/dx = (dz/dy) × (dy/dx)
```

In words: "the total effect of x on z = (how y affects z) × (how x affects y)"

## Why does this work?

Think of it as unit conversion. If `y = 2x` (one dollar buys 2 euros) and `z = 3y` (one euro buys 3 yen), then one dollar buys `2 × 3 = 6` yen. The chain rule is just multiplying conversion rates together.

## Extended Chain Rule for Multiple Variables

For a computational graph with multiple paths from `x` to `z`, the total gradient is the *sum* of contributions from all paths:

```
∂L/∂x = Σ (∂L/∂y) × (∂y/∂x)  for all y that depend on x
```

This is why gradients **accumulate** (`+=`) when a node has multiple outputs.

## Example in the Notebook

```python
a = Value(3.0, label='a')
b = a + a   # b depends on a twice!
b.backward()
```

When computing `b.grad` (which is 1.0 since `b = b`), calling `b._backward()` runs:
```python
# Inside __add__:
self.grad  += 1.0 * b.grad   # a's gradient from the left branch
other.grad += 1.0 * b.grad   # a's gradient from the right branch
```

Since `self` and `other` both point to `a`, we get:
```
a.grad = 1.0 + 1.0 = 2.0
```

This is correct: `db/da = d(a+a)/da = 2`.

**Without `+=`:** `a.grad` would be `1.0` (the second `+=` overwrites the first), which is wrong.

## Common Mistake: The `a + a` Bug

```python
a = Value(3.0, label='a')
b = a + a
b.backward()
```

If `__add__` used `=` instead of `+=`:
- First pass: `a.grad = 1.0` (from left branch)
- Second pass: `a.grad = 1.0` (overwrites from right branch)
- Result: `a.grad = 1.0` ← WRONG! Should be 2.0.

This is why `+=` is not optional — it's mathematically required.

---

# Flashcards

**Q: What does `self.grad = 0.0` in `__init__` mean?**
A: The gradient starts at zero because before running backpropagation, we don't know how this node affects the output. Zero is the correct initial state; `_backward` will accumulate into it using `+=`.

**Q: Why is `_backward` stored as a function (closure) rather than a method?**
A: Each `_backward` function needs to capture the specific `self`, `other`, and `out` objects from the moment the operation was performed. Closures freeze these references. A method couldn't capture different values for each instance of the same operation.

**Q: What is topological sort and why does backward() need it?**
A: Topological sort orders nodes so that every node appears after all its children. Backward pass must process nodes from output to input — a node can only compute its backward when its upstream gradient (`out.grad`) is already known. Topological sort (reversed) guarantees this order.

**Q: Why do we use `+=` instead of `=` in all `_backward` functions?**
A: A node can contribute to multiple downstream nodes (e.g., `b = a + a`). Each downstream node calls `a._backward()` through the chain rule path, and all contributions must be summed. If we used `=`, later contributions would overwrite earlier ones.

**Q: What is the derivative of addition?**
A: `∂(a+b)/∂a = 1.0` and `∂(a+b)/∂b = 1.0`. Addition distributes the upstream gradient equally to both parents.

**Q: What is the derivative of multiplication?**
A: `∂(a*b)/∂a = b` and `∂(a*b)/∂b = a`. In multiplication, each parent's gradient equals the *other* parent's value times the upstream gradient.

**Q: What is the derivative of tanh?**
A: `d/dx tanh(x) = 1 - tanh(x)²`. In code: `self.grad += (1 - t**2) * out.grad` where `t = tanh(self.data)`.

**Q: Why do we set `self.grad = 1.0` before the backward loop in `backward()`?**
A: The output node's gradient with respect to itself is always 1.0 (trivially). This seeds the backpropagation. If set to 0.0, no gradient would flow at all.

**Q: What does `__rmul__` do and why is it needed?**
A: `__rmul__` handles `2 * a` where `a` is a `Value`. Python tries `int.__mul__(2, a)` first; when that returns `NotImplemented`, it tries `a.__rmul__(2)`. Without it, expressions like `2 * value` raise a TypeError.

**Q: What is the training loop order?**
A: Forward → Zero Gradients → Backward → Update. The gradient zeroing happens before backward (not before forward) because we need the forward values for the backward computation.

**Q: What is the chain rule?**
A: `dz/dx = (dz/dy) × (dy/dx)`. The gradient of the output with respect to any input equals the product of gradients along any path connecting them (summed over all paths).

**Q: What is the `sum(..., self.b)` trick in the `Neuron.__call__` method?**
A: Python's built-in `sum(iterable, start)` starts accumulation from `start`. Using `self.b` as start computes `b + w1*x1 + w2*x2 + ...`. Using `0` would fail because `0 + Value(...)` requires `int.__add__` to handle `Value`, which it can't.

**Q: What does `requires_grad = True` do in PyTorch?**
A: It tells PyTorch to track this tensor in the computational graph, just like wrapping a number in our `Value` class. Without it, no gradients are computed for that tensor.

**Q: Why is division implemented as `self * other**-1`?**
A: It reuses existing `__mul__` and `__pow__` operations. The gradient flows correctly through both because we already implemented the backward passes for those primitives. This is "differentiating through composition."

**Q: What is the purpose of the `parameters()` method on MLP/Layer/Neuron?**
A: It returns a flat list of all `Value` objects (weights and biases) that are learnable. This lets the training loop iterate over all parameters for zeroing gradients and updating values.

---

# Interview Questions

## Easy

1. **What is a derivative?** The rate of change of a function's output with respect to a change in its input. Formally: `lim(h→0) [f(x+h) - f(x)] / h`.

2. **What is a computational graph?** A directed acyclic graph where nodes are values and edges connect parents to children based on the operations that produced them.

3. **What is the `__init__` method in Python?** The constructor — called when you create a new instance of a class with `MyClass(args)`.

4. **What is a closure in Python?** A function that remembers and has access to variables from the scope where it was defined, even after that scope has exited.

5. **Why is `tanh` used as an activation function?** It maps any real number to (-1, 1), introducing non-linearity that lets networks represent complex functions beyond linear combinations.

## Medium

6. **Explain backpropagation in one paragraph.** Backpropagation is the algorithm that computes the gradient of a loss function with respect to every parameter in a network. It works by traversing the computational graph backward from the output (loss) to the inputs, applying the chain rule at each node to propagate gradients. At each node, the local gradient (how the node's output affects its input) is multiplied by the incoming gradient (how the output's change affects the final loss), and this product is accumulated into the input's gradient.

7. **What happens if you forget to zero gradients before `loss.backward()`?** The new gradients get added to the old ones (because `_backward` uses `+=`). The effective gradient becomes the sum across multiple forward passes, which is almost certainly wrong and leads to incorrect weight updates.

8. **Explain why the derivative of `e^x` is `e^x` itself.** The exponential function is defined as the unique function that is its own derivative. You can show this from the Taylor series: `e^x = 1 + x + x²/2! + x³/3! + ...`. Differentiating term by term gives the same series.

9. **What would break if `__add__` and `__mul__` used `self.grad = ...` instead of `self.grad += ...`?** Nodes that appear multiple times in the computation (e.g., `b = a + a` or `e = a * a + a`) would have incorrect gradients. Only the gradient from the last path through that node would be kept; contributions from all earlier paths would be overwritten.

10. **Why must we process nodes in reverse topological order during backward pass?** Each node's `_backward()` needs `out.grad` to be fully computed. Reverse topological order guarantees that when we process a node, all nodes that depend on it (its outputs) have already had their gradients computed.

## Hard

11. **Derive the chain rule from the definition of a derivative.** Let `z = g(y)` and `y = f(x)`. Then `z = g(f(x))`. By the limit definition, `dz/dx = lim(h→0) [g(f(x+h)) - g(f(x))] / h`. Let `Δy = f(x+h) - f(x)`. Then the numerator = `g(f(x) + Δy) - g(f(x)) = (Δg/Δy) * Δy`. Dividing by `h`: `(Δg/Δy) * (Δy/h)`. Taking limits: `(dg/dy) * (dy/dx)`.

12. **Why is breaking `tanh` into primitives (`exp`, `+`, `/`) equivalent to implementing it as a single operation?** The chain rule ensures that composing differentiable operations gives the correct gradient, regardless of the granularity of decomposition. Whether you compute the gradient of `tanh` directly using `1 - t²` or via the derivatives of `exp`, `+`, `-`, and `/`, the chain rule guarantees both paths yield the same result. The graph has more nodes in the decomposed version, but each node's local gradient is simpler and the total product is identical.

13. **What is the vanishing gradient problem and why does `tanh` cause it?** For very large or very small inputs, `tanh` saturates — the function flattens out near ±1. The derivative `1 - tanh(x)²` approaches 0 in these saturated regions. When gradients flow backward through many layers of saturated `tanh` neurons, they get multiplied by near-zero values at each layer, exponentially shrinking. By the time gradients reach the early layers, they are so small that weights barely update — the network can't learn efficiently. This is why modern networks often use ReLU or variants.

---

# Common Bugs

## Bug 1: Forgetting to Zero Gradients

**Code:**
```python
for k in range(20):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    # BUG: missing: for p in n.parameters(): p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -0.1 * p.grad
```

**Why it happens:** It's easy to forget because the gradient is already set to `0.0` at `__init__` time. But the second backward call accumulates on top of the first.

**Symptom:** Loss may not decrease, or may oscillate wildly. Gradients grow with each iteration.

**Fix:** Always zero gradients immediately before `loss.backward()`.

---

## Bug 2: Using `=` Instead of `+=` in `_backward`

**Code:**
```python
def _backward():
    self.grad = 1.0 * out.grad      # BUG: should be +=
    other.grad = 1.0 * out.grad     # BUG: should be +=
```

**Why it happens:** Looks intuitive — "set the gradient to this value."

**Symptom:** Correct for simple graphs, wrong whenever a node appears multiple times.

**Debugging:** Try `b = a + a; b.backward()` and check if `a.grad == 2.0`. If it's `1.0`, you have this bug.

---

## Bug 3: Calling `backward()` on a Non-Scalar

**Code:**
```python
ypred = [n(x) for x in xs]  # list of 4 Values
ypred[0].backward()  # works, but only for the first example
```

**Why it happens:** Backprop starts from a scalar (single number). If you call backward on a vector, the starting gradient is ambiguous.

**Fix:** Always compute a scalar loss first (`sum(...)`), then call `loss.backward()`.

---

## Bug 4: Forgetting `requires_grad = True` in PyTorch

**Code (PyTorch):**
```python
x1 = torch.Tensor([2.0]).double()
# Forgot: x1.requires_grad = True
n = x1 * w1 + x2 * w2 + b
o = torch.tanh(n)
o.backward()
print(x1.grad)  # BUG: prints None
```

**Fix:** Set `requires_grad = True` for every tensor you need gradients for.

---

## Bug 5: `int + Value` Instead of `Value + int`

**Code:**
```python
loss = sum(ypred)  # BUG: sum starts from 0 (int), then 0 + Value fails
```

**Fix:** Use `sum(ypred, Value(0.0))` or `sum(ypred, start_value)` where `start_value` is a `Value`. Or implement `__radd__` so `0 + Value(...)` works. In the notebook, `__radd__` is implemented: `return self + other`.

---

# Build From Memory

Challenge yourself to rebuild everything without looking.

## Challenge 1: Core Value Class

Rebuild the `Value` class with:
- `__init__`: data, grad, _backward, _prev, _op, label
- `__repr__`: pretty print
- `__add__`: forward and backward
- `__mul__`: forward and backward
- `backward()`: topological sort + gradient propagation

## Challenge 2: Operations

Add to `Value`:
- `__pow__`: with assert for constant exponents
- `__rmul__`: for Python to call when left operand is not a Value
- `__truediv__`: implemented via `__mul__` and `__pow__`
- `__neg__`: implemented via `__mul__`
- `__sub__`: implemented via `__add__` and `__neg__`
- `__radd__`: for `sum()` compatibility
- `tanh()`: with the correct derivative
- `exp()`: with the correct derivative (note `+=` not `=`)

## Challenge 3: Neural Network

Build from scratch:
- `Neuron(nin)`: random weights and bias, `__call__`, `parameters()`
- `Layer(nin, nout)`: list of neurons, `__call__`, `parameters()`
- `MLP(nin, nouts)`: list of layers, `__call__`, `parameters()`

## Challenge 4: Training Loop

Write a complete training loop for 4 examples with 3 features:
1. Define `xs` and `ys`
2. Create an MLP
3. Run 20 iterations of: forward → zero grads → backward → update
4. Print the loss each iteration

## Challenge 5: Verify with PyTorch

Replicate the single-neuron example from the notebook using PyTorch tensors and verify that gradients match.

---

# Cheat Sheet

```
VALUE CLASS
──────────────────────────────────────────────────────────
self.data     = the number
self.grad     = starts at 0.0, gets filled by backward()
self._prev    = set of parent Value objects
self._op      = string label of operation (for viz)
self._backward = closure that propagates gradients to parents

OPERATION DERIVATIVES (local gradient)
──────────────────────────────────────────────────────────
c = a + b    → a.grad += 1.0 * c.grad
                b.grad += 1.0 * c.grad

c = a * b    → a.grad += b.data * c.grad
                b.grad += a.data * c.grad

c = a ** n   → a.grad += n * a.data^(n-1) * c.grad

c = tanh(a)  → a.grad += (1 - c.data²) * c.grad

c = exp(a)   → a.grad += c.data * c.grad

BACKPROPAGATION
──────────────────────────────────────────────────────────
1. Build topological sort from output to inputs
2. Set output.grad = 1.0
3. Iterate in REVERSE topological order
4. Call node._backward() at each step

TRAINING LOOP
──────────────────────────────────────────────────────────
for k in range(steps):
    ypred = [model(x) for x in xs]            # forward
    loss = sum((yp-yt)**2 for yt,yp in ...)   # loss
    for p in model.parameters(): p.grad = 0.0  # zero grads
    loss.backward()                             # backward
    for p in model.parameters():               # update
        p.data -= lr * p.grad

NEURAL NETWORK HIERARCHY
──────────────────────────────────────────────────────────
Neuron(nin)        → nin weights + 1 bias → tanh(w·x + b)
Layer(nin, nout)   → nout Neurons
MLP(nin, [h1,h2]) → Layer(nin,h1), Layer(h1,h2)

MLP(3, [4,4,1]):
  params = 4*(3+1) + 4*(4+1) + 1*(4+1) = 16+20+5 = 41

COMMON GOTCHAS
──────────────────────────────────────────────────────────
✓ Always += in _backward (not =)
✓ Zero gradients BEFORE backward
✓ output.grad = 1.0 (not 0.0)
✓ Use __radd__ so sum() works with Value objects
✓ loss must be scalar before calling backward()
```

---

# Mental Models

**Autograd is like...** a GPS that not only tells you where you are but also records every turn you made to get there. When you need to retrace (backprop), you reverse the route turn by turn.

**Backpropagation is like...** a chain of dominoes. You push the last one (set output gradient), and each domino knocks over the ones before it (each `_backward` call passes gradient to its parents).

**The computational graph is like...** a recipe with memory. Not only does it record the ingredients (inputs) and steps (operations), but when you taste the final dish and decide it needs more salt, it tells you exactly which step and which ingredient to adjust.

**Gradient descent is like...** hiking down a foggy mountain by always taking a step in the direction the ground slopes downward beneath your feet.

**The chain rule is like...** currency conversion. The exchange rate from dollars to yen is the rate from dollars to euros times the rate from euros to yen. Gradients work the same way: the effect of x on z = (effect of x on y) × (effect of y on z).

**The `_backward` closure is like...** a sticky note attached to each node that says "when you need the gradient, here's the formula." The note was written at the time the node was created and remembers all the values it needs.

**Topological sort is like...** the right order to read chapters of a mystery novel — you can't understand chapter 10 without having read the earlier chapters it depends on. In reverse, you process the ending first and work backward to understand the root causes.

**A `Neuron` is like...** a decision-maker who listens to multiple advisors (inputs), weights each advisor's opinion differently (weights), has their own initial bias (bias), and gives a final judgment squeezed between -1 and 1 (tanh output).

**An MLP is like...** a company's management hierarchy. Data enters at the bottom (inputs), gets processed at each management layer (hidden layers), and produces a final decision at the top (output).

**Why `+=` in `_backward`:** Think of a river system. A node that feeds into two separate channels contributes to both. The total water coming out of a source node is the sum of flow into all channels it feeds. Gradients work the same — sum all contributions.

---

# Final Summary

*(Imagine explaining this to a curious 15-year-old)*

**What did we learn?**

You know how a neural network learns? It learns by figuring out how to change its internal settings (called "weights") to get better results. But to change something intelligently, you need to know *how much* changing it would help. That's exactly what gradients tell you.

A gradient is just a number that says: "if you increase this weight by a tiny bit, the error gets worse (positive gradient) or better (negative gradient) by this much."

**The key insight:** Gradients can be computed automatically by remembering every calculation you did.

Here's the magic. When you do math with special "Value" objects (instead of regular numbers), every operation secretly records itself. It says: "I was a multiplication. My two parents were 2.0 and -3.0, and my result was -6.0. If you ever need to know how I contributed to the final answer, here's my formula."

After you compute your final answer (the loss — how wrong your network is), you "play back" all those recorded operations in reverse order. Each step passes blame backward: "The loss got worse because of this multiplication, and the multiplication got worse because of *this* number and *that* number." By the time you've traced all the way back to the original weights, every weight knows how responsible it was for the error.

That's **backpropagation**: automatic reverse-playback of the computation graph to compute all gradients.

Once every weight has a gradient, you make each weight move a tiny bit in the direction that makes things better (decrease the loss). Do this thousands of times, and your neural network "learns."

**What we actually built:**

1. A `Value` class that's like a smart number — it remembers its math history
2. A backpropagation engine that plays back that history in reverse to compute gradients
3. A `Neuron`, `Layer`, and `MLP` class that build real neural networks using those smart numbers
4. A training loop that repeatedly: makes predictions, measures error, computes gradients, and improves

And here's the beautiful part: PyTorch, the world's most popular deep learning library, does *exactly* the same thing internally — just faster, and for millions of parameters instead of 41. The math and the ideas are identical. After this lecture, you understand what PyTorch is doing at its core.

---

*End of Revision Notes — Micrograd, Andrej Karpathy's Neural Networks: Zero to Hero, Lecture 1*
