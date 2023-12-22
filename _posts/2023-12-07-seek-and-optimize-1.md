---
title: Seek & Optimize 1
author: haroun7
date: 2023-12-07 11:33:00 +0800
math: true
mermaid: true
---

Recent events since 2019 have already convinced me on [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html). I'm optimistic that we'll learn a more bitter lesson some day, perhaps sour and spicy lessons too. Until then, lets look at some classic algorithms. Why? Because we're now in the business of either scaling shit or discovering new tastes. Algorithms were built to scale and taste well before AI or ML were born. I'm sure there are lessons to be learnt. We'll repeat this process every now and then with new algorithms - in this first edition I asked a few co-workers for their favorite algorithms, here they are.

# Djikstra Algorithm
Good ol' [Djikstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#:~:text=Dijkstra's%20algorithm%20(%2F%CB%88da%C9%AA,Dijkstra's%20algorithm). It is a fast way to find the shortest path between nodes in a graph. While Djikstra's is the fastest known shortest path algorithm, it also has [Specialized variants](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Specialized_variants) for the case when weights are small integers bounded by a parameter $C$. The Tl;Dr of Djikstra is to maintain a few data structures: (1) `dist(node)`: represents distance from `source` vertex and is iteratively updated, (2) `prev(node)`: indicates the predecessor in the path to `node` from `source` and (3) `Q`: a list of nodes to still visit. `Q` is queried for smallest value per iteration - making it ideal for a min priority queue.

[`A*`](https://en.wikipedia.org/wiki/A*_search_algorithm) is sometimes seeen as an extension of Djiktra. While Djikstra produces "shortest paths tree" from a given source, A* only finds THE shortest path. `A*` as you may recall uses a heuristic to find the best path faster, and these heuristics need to be [admissible](https://en.wikipedia.org/wiki/Admissible_heuristic). If the heuristic is admissible and [`consistent`](https://en.wikipedia.org/wiki/Consistent_heuristic), then `A*` can terminate as soon as it finds the goal.

# Fenwick Tree
[First off, the Fenwick tree wasn't made by Fenwick. It was just described by Fenwick. It was originally proposed by Ryabko](https://en.wikipedia.org/wiki/Fenwick_tree). Thats kinda... sad. Anyway, the Ryabko tree is NOT a binary tree. It is a binary INDEX tree. Its good for prefix sums and it supports update (insert?) and computation. Each node in the tree is responsible to hold sum of a subset of indices of the original array. The subset is specifically the range of numbers from its parent's index to its own index. The last remaining question then is, what are the indices of nodes in the tree: "the parent of a node can be found by clearing its [`lsb``](https://en.wikipedia.org/wiki/Find_first_set)". Now the query and update algorithms are straightforward - use [`lso`] to navigate the implicit fenwick tree.

# Quicksort
Its cute in code:

```py
def quicksort(array: List) -> List:
    pivot_idx = 0
    if array:
        return (
            quicksort([x for x in array[pivot_idx + 1:] if x <= array[pivot_idx]])
            + array[pivot_idx]
            + quicksort([x for x in array[pivot_idx + 1:] if x > array[pivot_idx]])
        )
    else:
        return []
```

```ocaml
let rec quicksort = function
  | [] -> []
  | pivot :: tl ->
    let left, right = List.partition (fun x -> x < pivot) tl in
    quicksort left @ (pivot :: quicksort right);;
```

```cpp
// not entirely sure about this one :)
template<typename T>
std::vector<T> quicksort(std::vector<T> arr) {
    if (arr.size() <= 1) return arr;
    auto pivot = arr[arr.size() / 2];
    auto [left, middle, right] = [&] {
        std::vector<T> l, m, r;
        for (const auto &e : arr) (e < pivot ? l : (e == pivot ? m : r)).push_back(e);
        return std::make_tuple(quicksort(l), m, quicksort(r));
    }();
    left.insert(left.end(), middle.begin(), middle.end());
    left.insert(left.end(), right.begin(), right.end());
    return left;
}
```

Need we say more?

# SDM
[Supervised Descent Method](https://www.ri.cmu.edu/pub_files/2013/5/main.pdf) is an optimization algorithm for non-linear least squares (NLS) function. Its tailor-made for computer vision, where a lot of problems can be formulated as NLS problems. Critically, using Newton's method requires Hessians which can't be computed in _traditional_ CV - for example, we think of [SIFT features](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) as non differentiable... SDM works around this by learning descent directions from training data. Its a neat little idea and implementation isn't too bad - [there's a small python repo](https://github.com/Ning-Ding/SDM/tree/master) that implements it (I haven't actually used it, so ymmv).

# Fibonacci Search

[Fibonacci search](https://en.wikipedia.org/wiki/Fibonacci_search_technique) is a search algorithm. Before we discuss it, lets briefly discuss search algorithms. However, before we discuss algorithms, we have to discuss problems. Theoretical CS teaches us about [computational problems](https://en.wikipedia.org/wiki/Computational_problem) and its various types. Search problems and optimization problems are merely two such problems. A search problem can be defined by a relation between input and solution. Meanwhile optimization problems are represented by some kind of objective function (and constraints). A search algorithms given an input will find a solution that is in the relation implied by the search problem. Now, according to [this stackoverflow post](https://cs.stackexchange.com/questions/96567/search-problem-vs-optimization-problem), search algorithms can be reduced to optimization algorithms? Anyway, an interesting idea proposed on the stackoverflow was is that the idea of `verification` is the primary difference between search and optimization - proposed solutions to optimization problems are hard to verify correctness while search problems are straightforward to verify (assuming the relation can be computed.)

Oddly enough we treat ML as an optimization problem. However, isn't it ... easy to verify? Once we have a candidate model, we just run it on the test set and see if its good? However, `generation` seems to be harder to verify - we'd need to verify that the outputs are sensible everywhere, I guess?

Eitherway, search problems are solved using search algorithms and we often see three types of search algorithms: linear, binary and hashing. Fibonacci Search is a divide-and-conquer search which instead of splitting space into two equal halves, splits it into spaces whose sizes are consecutive fibonacci numbers. Apparently this allows us to conduct a search using just addition and subtraction. This was important when this algorithm was published (1960).

# Grover's Algorithm

[This quantum search algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm) is an unstructured search which finds an input to a function that leads to the desired output in $\sqrt{N}$ function calls. The formulation of the problem is that if there is a criterion $f$ that we're trying to satisfy (and suppose $w \in X$ satisfies $f$ - i.e., $f(w)=1$), we can write that as a unary operator:

$$
U_w  \ket{x} = (-1)^{f(x)} \ket{x}
$$

This unary operator $U_w$ is called an oracle. The algorithm to find $w$ is given on [wikipedia](https://en.wikipedia.org/wiki/Grover%27s_algorithm#Algorithm), and it essentially takes a state space (probability distribution), applies the oracle operator, and then applies a diffusion operation. Then we repeat this a bunch of times, and the bunch of times happens to be $O(\sqrt{N})$. The really cute bit seems to be the proof of correctness - the oracle operator and diffusion operator are both reflections which when applied one after the other bring the state vector closer to the correct answer $w$ but keeps it in the plane spanned by uniform distribution and $w$. (PS: my terminology is casual, please don't kill me, physicists.)

------

**Acknowledgements:** Thank you, Tong, Animesh, Peihong and Sidi for your inputs on algorithms :)