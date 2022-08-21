# <i class="fas fa-book fa-fw"></i> Linear Regression

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

```{video} ../../_static/manim_animations/linear_regression/data_anim.mp4
:width: 100%
```

When using machine learning, we generally make an assumption that there is a relationship between the input and the target (i.e., square footage of the house and its sale price). We are going to say that there exists some secret (unknown) function $f$ such that the price of a house is approximately equal to the function's output for the houses input data.

```{video} ../../_static/manim_animations/linear_regression/true_function_anim.mp4
:width: 100%
```

Note that we really do need the qualifier "approximately" above. We are not saying that the output has to be exactly equal, but rather that it is close. The reason we allow for this wiggle-room is that we are allowing for the fact that our model of the world might be slightly wrong. There are probably more factors outside the square footage that affect a house's price, so we are never hoping to find an exact relationship between this input and our output; just one that is "good enough". Alternatively, another reason to only need approximately equal for our model is to allow for the fact that there might be uncertainty in the process of measuring the input (square footage) or output (price).

```{margin}
1\. 📝 *Notation*: When we subscript a variable like $x_i$, it means we are talking about the $i^{th}$ example from our given dataset. In our example, when we say $x_{12}$, we are talking about the 12th house in the dataset we were given.

When we use $x$ without subscripts, we are talking about any input from our domain. In our example, when we say $x$, I mean some arbitrary house.
```

To be a bit more precise, we will specify how we believe "approximately equal" works for this scenario. Another term for "the assumptions we make" is the **model** we are using. In this example, we will use a very common model about how the input and target relate. We first show the formula, and then explain the parts (1).

```{margin}
2\. 📝 *Notation*: $\mathbb{E}\left[X\right]$ is the expected value of a random variable (the "average" outcome). See more [here](https://www.investopedia.com/terms/e/expected-value.asp).
```

The way to read this formula above is to say that the outcomes we saw, $y_i$, come from the true function $f$ being applied to the input data $x_i$, but that there was some noise $\varepsilon_i$ added to it from some source of error. We also will make an assumption about how the $\varepsilon_i$ as well: we assume that $\mathbb{E}\left[\varepsilon_i\right] = 0$, which means on average, we expect the noise to average out to 0 (i.e., it's not biased to be positive or negative). The animation below shows a visual representation of this model (2)


```{video} ../../_static/manim_animations/linear_regression/model_anim.mp4
:width: 100%
```

Earlier, we said a common goal for machine learning is to make predictions about future data. Since we won't necessarily be given the correct output ahead of time, we often focus on trying to learn what the true function $f$ is. So commonly we try to estimate $f$ from the data we are provided, and then use this learned function to make predictions about the future. You might be able to tell what makes this challenging: we don't know what $f$ is! We only have access to this (noisy) data of inputs/outputs and will have to somehow try to approximate $f$ from just the given data.

```{margin}
3.\ 📝 Notation: A  $\hat{\ }$ in math notation almost always means "estimate". In other words, $\hat{f}$​ is our best estimate of this unknown function $f$. What do you think $\hat{y}$​ is supposed to represent?
```

To phrase this challenge mathematically, a common goal in machine learning is to learn a function (3) $\hat{f}$​ from the data that approximates $f$ as best as we can. We can then use this $\hat{f}$​ to make predictions about new data by evaluating $\hat{y} = \hat{f}(x)$. In English, for a given example ($x$), we are predicting what we think the label should be ($\hat{y}$) based on this function we *think* is a good estimate ($\hat{f}$) of the unknown true function ($f$). It's likely our estimate won't be exactly correct, but our hope is to get one that is as close to this unknown truth as possible. We will come back to *how* we estimate this function later, but it has something to do with finding a function that closely matches the data were given.

```{video} ../../_static/manim_animations/linear_regression/predictor_anim.mp4
:width: 100%
```

Since we can't actual observe the true function $f$, assessing how good a potential $\hat{f}$ is will be quite challenging. The animation above previews how we will do this by comparing how $\hat{f}$ does on the data we trained it from. More on this in the next section.

## ML Pipeline

Machine learning is a broad field with many algorithms and techniques that solve different parts of a learning task. We provide a generic framework of most machine learning systems that we call the **ML Pipeline**, which is shown in the image below. Whenever we are learning a new topic, try to identify which part of the pipeline it works in.

TODO ML Pipeline image. TODO add step after hat_y to evaluate the model.
