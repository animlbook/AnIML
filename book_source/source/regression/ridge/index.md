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
    ref_index: 1
---

<!-- Everything below is a copy of an earlier chapter for reference -->

# <i class="fas fa-book fa-fw"></i> Regularization: Ridge

One of the central goals of machine learning is to make predictions about the future using data you have collected from the past. Machine learning is particularly effective when you have large amounts of data that allow the machine to automatically learn the patterns of interest from the data itself.

For example, say we are thinking about selling our house and we want to predict how much it will sell for based on the information about the house (e.g., how big it is, how many bathrooms, if there is a garage, etc.). Instead of trying to write out a program to determine the price by hand, we will give historical data to the computer and let it learn the pattern.

The most crucial thing in machine learning is the data you give it to learn from. A popular saying amongst machine learning practitioners goes **"Garbage in, Garbage out"**. So before we actually talk about how to make a machine learn, we need to talk about data and the assumptions we will make about the world.

Sticking with the housing example our goal will be to predict how much our house will sell for by using data from previous house-sales in neighborhoods similar to mine. We'll suppose we have a dataset with this information that has examples of $n$ houses and what they sold for.

$$
\begin{aligned}
    (x_1, y_1) &= (2318\ \text{sq.ft.}, \$315\text{k})\\
    (x_2, y_2) &= (1985\ \text{sq.ft.}, \$295\text{k})\\
    (x_3, y_3) &= (2861\ \text{sq.ft.}, \$370\text{k})\\
    \ &\vdots \\
    (x_n, y_n) &= (2055\ \text{sq.ft.}, \$320\text{k})\\
\end{aligned}
$$

The way we represent our data is a $n$ input/output pairs where we use the variable $x$ to represent the input and $y$ to be the output. Each example in our dataset will have **input data**, represented with the variable $x$. In our context of housing prices, there is one data input for each house (the square footage), but in other contexts, we will see that we are allowed to have multiple data inputs. The outcome for the house is its sale price, and we use the variable $y$ to represent that. Do note that this $y$ variable generally goes by many names such as **outcome/response/target/label/dependent variable**.

It is sometimes helpful to visualize the relationship between input and output. Visually, we could plot these points on a graph to see if there is a relationship between the input and the target.

```{video} ../../_static/regression/linear_regression/manim_animations/data_anim.mp4
```

When using machine learning, we generally make an assumption that there is a relationship between the input and the target (i.e., square footage of the house and its sale price). We are going to say that there exists some secret (unknown) function $f$ such that the price of a house is approximately equal to the function's output for the houses input data.

```{video} ../../_static/regression/linear_regression/manim_animations/true_function_anim.mp4
```

Note that we really do need the qualifier "approximately" above. We are not saying that the output has to be exactly equal, but rather that it is close. The reason we allow for this wiggle-room is that we are allowing for the fact that our model of the world might be slightly wrong. There are probably more factors outside the square footage that affect a house's price, so we are never hoping to find an exact relationship between this input and our output; just one that is "good enough". Alternatively, another reason to only need approximately equal for our model is to allow for the fact that there might be uncertainty in the process of measuring the input (square footage) or output (price).

```{margin}
{{ref_index}}\. 📝 *Notation*: When we subscript a variable like $x_i$, it means we are talking about the $i^{th}$ example from our given dataset. In our example, when we say $x_{12}$, we are talking about the 12th house in the dataset we were given.

When we use $x$ without subscripts, we are talking about any input from our domain. In our example, when we say $x$, we mean some arbitrary house.
```
