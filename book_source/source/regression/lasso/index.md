---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_dna: 1
    ref_all_subsets_strs: 2
---

```{code-cell} ipython3
:tags: [remove-input]

# If you are reading this notebook on a Binder, make sure to right-click the file name in
# the left-side file viewer and select `Open With > Notebook` to view this as a notebook
```

# <i class="fas fa-book fa-fw"></i> Regularization: LASSO


In the last chapter on {doc}`../ridge/index` we discussed one technique to prevent our models from overfitting by using the concept of *regularization* to modify the quality metric used when learning our model parameters to penalize predictors with large coefficients. We augmented our original training quality metric that only cared about the training error $L(w)$ (in our case $MSE(w)$) by adding in a regularizer $R(w)$ to measure the magnitude of our models coefficients. The $\lambda$ term was a hyperparameter to control for how much we want to penalize high coefficients.

$$\hat{w} = \argmin{w}L(w) + \lambda R(w)$$

We also briefly discussed how there are many ways to define this regularizer $R(w)$. In the last chapter on {doc}`../ridge/index` we defined $R(w) = \lVert w \rVert_2^2 = \sum_{j=1}^D w_j^2$ which yields the Ridge Regression model. We also described using the regularizer $R(w) = \lVert w \rVert_1 = \sum_{j=1}^D \lvert w \rvert$. In this chapter we will discuss the importance of **feature selection** and how we can use the LASSO to help us accomplish that goal.

## Feature Selection

The process of **feature selection** is to narrow down from our list of possible features some subset of them that are the most "meaningful" or the most "effective" at predicting our labels. There are a few reasons why we might care to select just a few features to work with:

```{margin}
{{ref_dna}}\. Consider a simple approach to try learning on DNA sequences (strings of nucleotides A, C, T, G). You could imagine if you use one feature per section of DNA, we would have something like a $\hat{w}$ with approximately 3.2 billion coefficients that need to be learned ([source](https://nigms.nih.gov/education/Inside-Life-Science/Pages/Genetics-by-the-Numbers.aspx#:~:text=6&text=That's%20how%20many%20feet%20long,round%20trips%20to%20the%20Moon.))!
```

* *Model Complexity*: Models with more features tend to be more complex and are more likely to overfit. By reducing the numbers of features to a smaller subset that are the most meaningful, we hopefully can avoid overfitting.
* *Interpretability*: while a bit more nebulous in nature, we have a strong intuition that simpler models are easier for ML practitioners and users to understand. The idea is that if a model is simpler, it requires less cognitive load for a human to understand what the model is doing or why it makes the decisions it does.
* *Computational Efficiency*: In many settings, we can have feature spaces that are extremely large<sup>{{ref_dna}}</sup>. If there are many features, there will be many coefficients we have to learn. This will both cause a slow down in training time since we have to update and compute information for more parameters, but also when it comes to making predictions for new values. If we have to evaluate $\hat{y} = \hat{w}^Th(x)$, this requires doing an element-wise product and then summing up terms for every feature!

However, many of these problems are solved if we demand that the learned coefficients $\hat{w}$ are **sparse**, or many of the coefficients are 0. That means we only have to consider the relatively few coefficients that are non-zero. How many is enough is yet another modeling decision we will set as a hyperparameter.

$$\hat{y} = \sum_{\hat{w}_j \neq 0}\hat{w}_j h_j(x)$$

This notion is also helpful if you consider a real-world setting such as our task of predicting the price of houses. There are potentially many features that we could use, but maybe only a few of them are the "most important bits" of information that have the highest impact on house price. Additionally, many of the features might be strongly correlated so including them all might be redundant.

Sparsity in our models is also a useful tool for scientific discovery. Imagine a neurologist investigating how brain activity relates to happiness. The inputs $x$ are descriptions of an image scan of the brain (many features for all pixels in the image) and the output is some number from 0 to 10 describing how happy the person was at the time of that brain scan. If we train a model to predict happiness from brain state, and additionally require that model is *sparse*, it might help us find the most important areas of the brain for influencing happiness. This can then inspire future research into brain activity.

```{image} ./brain.png
:align: center
:alt: An image of a brain activity lighting up (inputs x) and mapping on a scale from 0 to 10
```

## All Subsets

Our initial approach to selecting the subset of features will, in some sense, be the theoretically "best" way to find the best subset of features, but we will see might not actually work in practice.

If we are interested in trying to find the subset of features that are the most important, the simplest approach we could consider is just try out *every* subset of features to find which one is optimal. As a small example, if you have features A and B, you could try out models that use
* None of the features (i.e., the empty set of features)
* Just feature A
* Just feature B
* Both feature A and B

With more features to choose from, you can do a similar enumeration of all possible subsets. From our discussion of model complexity in {doc}`../assessing_performance/index`, we can already start to make some assessments of how using more or fewer features in our subset of selected feature affects our model performance:

* If we select fewer features, our model will tend to be simpler (higher bias, lower variance). If it is too simple, we will expect both high training error and high true error.
* If we select more features, our model will tend to be more complex (lower bias, higher variance). If the model is too complex, we will expect low training error and high true error.

So, if we want to choose the right subset of feature of the right size, we will have to use a procedure like we have discussed multiple times previously: Try all of the options and compare them by some sort of validation error (either using a validation set or cross validation).

Consider our housing dataset, and only considering the features:
* \# bathrooms
* \# bedrooms
* sq.ft. living room
* sq.ft. lot
* \# floors
* year built
* year renovated
* waterfront

```{margin}
TODO add animation
```

The following graph shows running this experiment on all subsets of these features. The x-axis shows the number of features considered at a time and the y-axis shows the validation $MSE$ on some held out dataset. Each dot corresponds to one model trained by some subset of these features. For example one subset of these features of size 3 is `[\# bathrooms, sq. ft. living, year built]`. Even for a fixed size for the subset of features (e.g., 3), you will see performance all over the place from different subsets of that size simply because some subsets of feature are more informative than others. For each size, the optimal subset (based on minimum validation error) is highlighted in pink. So using our selection process, we would probably choose the features associated to the pink dot with 6 features chosen.

```{image} ./all_subsets.png
:align: center
:alt: A graph showing the errors of models trained on various subsets of the features. Described in last paragraph.
```

One important note is that the optimal set of $k$ features may or may not have overlap with the optimal set of $k+1$ or $k-1$ features. In other words, the sets of optimal features are not "nested". For example, it's possible for the optimal subset of features, when limited to one feature, is just `[sq.ft. lot]`. But when we allow it to include 2 features, maybe it's the case that `[\# bathrooms, \# bedrooms]` are more information *jointly* but not separately. Because of this, we are really limited to trying all possible subsets if we want to find the globally optimal subset of features.

### Efficiency of All Subsets

One question we might ask is: How efficient is this algorithm? If I have $D$ features to choose from, how much time might it take to try out all of these possibilities? We'll see that this algorithm is actually not that practical because its runtime is primarily based on the number of subsets there are of $D$ features, and it turns out that number can be high for even a moderate number of features.

To count the number of possible subsets, we could try writing out all of the possible models (for consistency, we will use the notation $w_i$ to match the index $i$ for our feature vector $h_i(x)$.

| Model                                                          |
| :------------------------------------------------------------- |
| $\hat{y}_i = w_0$                                              |
| $\hat{y}_i = w_0 + w_1h_1(x)$                                  |
| $\hat{y}_i = w_0 + w_2h_2(x)$                                  |
| ...                                                            |
| $\hat{y}_i = w_0 + w_1h_1(x) + w_2h_2(x)$                      |
| ...                                                            |
| $\hat{y}_i = w_0 + w_1h_1(x) + w_2h_2(x) + ... + w_Dh_D(x)$    |


```{margin}
{{ref_all_subsets_strs}}\. We are assuming here that you always include the intercept $w_0$. You could extend all of this discussion to include/exclude the intercept as well if you choose.
```

Even with just our $D = 8$ features, there are quite a lot of them to list out. But we can actually come up with a trick to count them without listing them out! We make the observation that every one of these models either contain some feature $h_i(x)$ or they do not. In other words, we can write out a string of 0s and 1s to name each of these models, where a 1 at index $i$ indicates feature $i$ is in the model, and a 0 indicates it's not<sup>{{ref_all_subsets_strs}}</sup>. So we could write these model strings down as following

| Model                                                          | String              |
| :------------------------------------------------------------- | :------------------ |
| $\hat{y}_i = w_0$                                              | `[0 0 0 ... 0 0 0]` |
| $\hat{y}_i = w_0 + w_1h_1(x)$                                  | `[1 0 0 ... 0 0 0]` |
| $\hat{y}_i = w_0 + w_2h_2(x)$                                  | `[0 1 0 ... 0 0 0]` |
| ...                                                            | ...                 |
| $\hat{y}_i = w_0 + w_1h_1(x) + w_2h_2(x)$                      | `[1 1 0 ... 0 0 0]` |
| ...                                                            | ...                 |
| $\hat{y}_i = w_0 + w_1h_1(x) + w_2h_2(x) + ... + w_Dh_D(x)$    | `[1 1 1 ... 1 1 1]` |

You may or may not have seen how to count all binary strings (strings of 1s and 0s) of length $D$ before. The trick is to notice that at each index, there are two choices (0 or 1). So the number of strings possible is $2 \cdot 2 \cdot 2 \cdot ... \cdot 2$ ($D$ times), also written as $2^D$. Computer scientists use the notation $\mathcal{O}(2^D)$ to describe the runtime of this algorithm primarily depends on the number of subsets here $2^D$.

If you haven't seen algorithms that scale like this before, you might be thinking that it can't be *that* bad. Let's work through a concrete example to see how this model scales. Suppose it takes our computer 8 minutes to calculate all of the validation errors for all subsets of 8 features (that works out to about 20 milliseconds per subset of features). Take a guess how long you think it would take to train the following all subset models before expanding to see their answers. Really, make a guess before clicking the drop down to see how close you were!

```{dropdown} What if we tried all subsets of 16 features?
This would take about $21$ **_minutes_** (about 256 times as long as 8 features)
```

```{dropdown} What if we tried all subsets of 32 features?
This would take about $3$ **_years_** (about 17 million times as long as 8 features)
```

```{dropdown} What if we tried all subsets of 100 features?
This would take about $7.5 \cdot 10^{20}$ **_years_** (a lot longer than our 8 features model). For reference, this is approximately 50,000,000,000 times the age of our universe 😱
```

To be very clear, 100 features is not what companies are all excited about when they are talking about "Big Data". The datasets used in many real-world applications are orders of magnitudes larger than these relatively small examples. So clearly, this algorithm while it will find the best subset, it won't find it in a practical timeline!

So if this algorithm is not practical, what can we do? At a high level, we will have to settle for an approximation. We will come up with some algorithm that will run in our lifetimes, but it won't guarantee to find the best possible subset.

At a high level, the two approximations we will explore for the rest of this chapter are:

1. Greedy algorithms
2. Regularization

## Greedy Algorithms for Feature Selection

A **greedy algorithm** is a flavor of algorithms for approximating intractable solutions like the best subset of features. So instead of finding the best option by trying every possible option, we will build up a solution by *greedily* choosing the best option at the time. Many of us are familiar with greedy algorithms. Every time to you go to a new grocery store, you often just start shopping down the aisles right in front of you rather than getting a floor plan of the whole store to plan some globally optimal route. It's possible your greedy route might not be optimal, but it's hopefully not that much worse and maybe saves you time that it would have taken you to find the map and compute the "optimal" path.

There are lots of examples of algorithms that try to approximate solutions using a greedy approach. For feature selection, they generally fall into one of three possible types, although others exist as well!

* **Forward Stepwise** algorithms generally start with an empty set of important features, and iteratively add features to this set as performance improves.
* **Backward Stepwise** algorithms generally start with the full set of features as important features, and iteratively remove features that are the least important.
* **Combining these two options** algorithms generally have some heuristic of combining these two approaches of adding/removing features from this selected set.

Importantly, regardless of which type of algorithm you use, the answer it will find is still an *approximation*. There are no guarantees that it is going to find the optimal subset of features; as we discussed doing so will take to long.

We will focus on the Forward Stepwise algorithm as a concrete example, and leave it as an exercise to explore how it would be adapted for a backwards or combination version.

### Example: Forward Stepwise Algorithm

The high level idea of the Forward Stepwise algorithm is to iteratively add features to our selected set as they improve performance. We assume you pre-select some desired number of features $k$ as the maximally allowed size of the set of important features; although we have in this algorithm to select a smaller subset if validation performance starts to decrease.

```{prf:algorithm} Forward Stepwise Algorithm
:label: forward-stepwise

**Inputs**
* A training dataset $X_{train} \in \mathbb{R}^{n\times D}$ with features $h_1(x), h_2(x), ..., h_D(x)$
* A validation error function $MSE_{val}(\hat{f})$
* A maximum number of selected features $k$

**Output** A set of selected features $S \subseteq \{h_1(x), h_2(x), ..., h_D(x)\}$ with $\lvert S \rvert \leq k$

1. $\texttt{min_val} \gets \infty$
2. $S_0\ \gets \emptyset$
3. for $i \gets [1, k]$:
    1. Find feature $h_j(x)$ not in $S_{i-1}$, that when combined with the features in $S_{i-1}$ minimize the validation loss $MSE_{val}(\hat{f})$ the most. Call this model $\hat{f}_i$ (trained on $h_j(x)$ and $S_{i-1}$)
    2. if $MSE_{val}(\hat{f}_i) > \texttt{min_val}$
        1. Break and return $S_{i-1}$
    3. $S_i \gets S_{i-1} \cup \{h_j(x)\}$
    4. $\texttt{min_val} \gets MSE_{val}(\hat{f}_i)$
4. Return $S_i$
```

While the details are a bit complicated, this is really just saying "keep adding features to our set of important features until we reach $k$ features or the validation error increases".

```{admonition} Practice
:class: important

Suppose we were working with a small set of 4 features on our house price prediction task. Below we show two tables, the first shows the validation errors when considering just a single feature and the second shows the validation errors considering two features. **Following this Forward Stepwise Algorithm, which subset of two features would be selected by this algorithm?**

Table 1: Validation Errors for Subsets of Size 1

| Feature    | Validation Loss |
| :--------- | --------------: |
| # bath     | 201             |
| # bed      | 300             |
| sq ft      | 157             |
| year built | 224             |

Table 2: Validation Errors for Subsets of Size 2

| Features (unordered) | Validation Loss |
| :------------------- | --------------: |
| # bath, # bed        | 120             |
| # bath, sq ft        | 130             |
| # bath, year built   | 190             |
| # bed, sq ft         | 137             |
| # bed, year built    | 209             |
| sq ft, year built    | 145             |
```

```{code-cell} ipython3
:tags: ["remove-input"]

questions = [
    {
        "question": r"""Which set of features would be chosen following this Forward Stepwise model?""",
        "type": "multiple_choice",
        "answers": [
            {
                "answer": "# bath, # bed",
                "correct": False,
                "feedback": "Not quite! While this is the best subset with two features, is this the one that our forward algorithm would end up picking?"
            },
            {
                "answer": "# bath, sq ft",
                "correct": True,
                "feedback": "Correct! The first iteration would choose sq ft because it was the single feature with lowest validation error. Then on our second iteration, we will only consider adding one of the other unchosen features to be included with sq ft. The set of features that includes sq ft that minimizes validation error is # bath and sq ft. Note that this algorithm did not find the globally optimal subset of features of # bed and # bath"
            },
            {
                "answer": "# bath, year built",
                "correct": False,
                "feedback": "Not quite!"
            },
            {
                "answer": "# bed, sq ft",
                "correct": False,
                "feedback": "Not quite!"
            },
            {
                "answer": "# bed, year built",
                "correct": False,
                "feedback": "Not quite!"
            },
            {
                "answer": "sq ft, year built",
                "correct": False,
                "feedback": "Not quite!"
            },

        ]
    },
]

from jupyterquiz import display_quiz
display_quiz(questions, shuffle_answers=False)
```