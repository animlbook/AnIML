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
    ref_pipeline: 1
    ref_coefficients: 2
    ref_quality_metric: 3
    ref_norm_soon: 4
---

# <i class="fas fa-book fa-fw"></i> Regularization: Ridge

In the last chapter on {doc}`../assessing_performance/index` we introduced the concept of *overfitting* and the importance of selecting a model of the right complexity. Our goal shifted from our first attempt at simply minimizing training error, to trying to find a model that would perform best in the future, approximated by our test error. Since we want the test error to be an unbiased estimate of the future, we had to use techniques like a validation set or cross validation to choose the best model complexity, and then test on our test set to estimate our chosen model's performance.

The major pattern we saw was that using a model that was too complex, like an extremely high degree polynomial, it had no option but to overfit if it was sufficiently complex for the data it was trained on. But what if we could change our model so that it was aware when it was overfitting and prevent that from happening? In other words, what if the model could *self-regulate* to prevent itself from overfitting. This idea of modifying the model to prevent itself from overfitting leads us to a set of techniques known as **regularization**. We will explore the ideas behind two of the most most popular regularization techniques, introducing the the background and one of the techniques in this chapter and the other in the next (TODO link).

## Background: Coefficients

rrom the last chapter, we have a good understanding that learned predictors that bounce around wildly like a high-degree polynomial on a small amount of data seem to be overfit. For example, if we train a 30 degree polynomial on 10 examples, we could visually see this level of overfitting in the predictions made by the model.

```{code-cell} ipython3
:tags: [hide-input]

# This code cell defines a helper function we use below.
# Understanding the code in this hidden cell is not important

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Special functions for Jupyter Notebook output
from IPython.display import display, Markdown

LEGEND_PARAMS = {"facecolor": "white", "framealpha": 1}

sns.set()

# Set random seet
np.random.seed(123)


def true_f(x: float) -> float:
    """
    True function f(x) (unknown to us)
    """
    return 2 * x - 1.4 * x ** 2


def animl_generate_data(n: int) -> tuple[np.array, np.array]:
    """
    Creates and returns a DataFrame with n randomly drawn examples
    """
    Xs = np.random.uniform(0, 1, n)                # generate n numbers uniform at random from [0, 1]
    ys = true_f(Xs) + np.random.normal(0, 0.1, n)  # evaluate f at each x and add Gaussian noise (mean=0, variance=.1)
    # Need to make Xs a 2D array (with 1 column)
    return Xs.reshape(-1, 1), ys


def animl_plot_data(data: pd.DataFrame, x: str, y: str) -> None:
    fig, ax = plt.subplots(1)
    sns.scatterplot(data=data, x=x, y=y, color="black", alpha=0.9, ax=ax, label="Data")

    ax.set_title("Relating Living Room Square Footage to House Price")
    ax.legend(**LEGEND_PARAMS)


def animl_plot_regression(Xs: np.array, targets: np.array,
                          model: LinearRegression) -> None:
    fig, ax = plt.subplots(1)

    # Plot predictor
    min_x, max_x = Xs.min(), Xs.max()
    plot_xs = np.linspace(min_x, max_x, num=100)
    pred_ys = model.predict(plot_xs.reshape(-1, 1))
    ax.plot(plot_xs, pred_ys, color="#59C5E6", linewidth=2, label="Predictor")

    # Plot data
    sns.scatterplot(x=Xs[:, 0], y=targets, ax=ax, color="black", label="Data")

    # Customize plot
    y_min, y_max = targets.min(), targets.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax.set_title("Predicting Price (dollars) with Living Room size (sq. ft.)")
    ax.legend(**LEGEND_PARAMS)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    align: center
---

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def overfitting_experiment(n: int, degree: int, plot: bool=True) -> LinearRegression:
    """
    Runs an experiment generating a polynomial model trained on
    n points generated randomly. Plots the model predictions compared
    to the training data. Returns the model afterward.
    """

    Xs, ys = animl_generate_data(n)

    # We use a new construct from scikit-learn called Pipeline to make
    # a sequence of preproccessing steps. First to transform the data to
    # have polynomial features, then to train a linear regression model.
    # We can call fit/predict on this pipeline and it runs each step
    # in sequence (first transform features, then to linear regression model)
    model = Pipeline([
        ("features", PolynomialFeatures(degree)),
        ("linear model", LinearRegression())
    ])
    model.fit(Xs, ys)

    # Plot predictions
    if plot:
        animl_plot_regression(Xs, ys, model)
    return model

model_small_data = overfitting_experiment(n=10, degree=30)
```

However, if we instead had a lot more data we don't see the same overfitting behavior visually even with a model of the same complexity.

```{code-cell} ipython3
---
mystnb:
  figure:
    align: center
---

model_large_data = overfitting_experiment(n=500, degree=30)
```

```{margin}
{{ref_coefficients}}\. Note this is a general statement and not saying the coefficients strictly increase. You can see in this graph that the magnitude of the largest coefficients isn't monotonically increasing. This is for a few reasons:

1. Numbers in Python can only be *so large* (finite memory)
2. As we increase the degree of feature polynomials, the numbers of these features is quite large so the coefficients don't have to be quite as large to compensate (more on this later)

But this trend is still generally true, that for these high degree polynomials, we tend to get large coefficients when they are overfitting. Some of these coefficients are orders of magnitude larger than the coefficients of the simple, linear models.
```

So while we can easily spot the differences in these models with our eyes, there is another method for being able to spot overfitting. It turns out that the *magnitude of the coefficients* for these very "wiggly" functions, tend to be quite large. In other words, this wiggling happens generally because the learned coefficients tend to be very large in magnitude. For example, look at what happens to the highest magnitude (largest in absolute value) coefficients for our polynomial regression model, as we increase the polynomial features<sup>{{ref_coefficients}}.

```{code-cell} ipython3
---
mystnb:
  figure:
    align: center
---

MAX_DEGREE = 30

Xs, ys = animl_generate_data(10)

degrees = list(range(1, MAX_DEGREE + 1))
largest_coeffs = []
for p in degrees:
    # Generate data with polynomial features of degree p
    poly_features = PolynomialFeatures(p)
    X_poly = poly_features.fit_transform(Xs)

    # Train the model
    model = LinearRegression()
    model.fit(X_poly, ys)

    # Find largest coefficient
    coeff = np.abs(model.coef_).max()
    largest_coeffs.append(coeff)

# Plot the coefficients
plt.plot(degrees, largest_coeffs)
plt.title('Overfitting -> Large Coefficients')
plt.xlabel('Degree')
plt.ylabel('Magnitude of Largest Coefficient (log)')
plt.yscale('log')
```

So in short, having coefficients with very large magnitudes is a symptom that our model might be overfitting. So one idea to prevent overfitting, is to control the magnitude of these coefficients to make sure they aren't too large! The major idea of **regularization** is changing our model to be aware and stop itself from making its coefficients too large.

## Regularization

TODO ML pipeline image

!!!tip
    Big Idea: Change our *quality metric* to increase the cost of a model when its coefficients are getting large

Before, we trained our regression model by finding the parameters that minimized some loss function $L(w)$ (e..g, $MSE(w)$). The idea here was by minimizing the error on our training set, we were hopefully find the "best" model.

$$\hat{w} = \argmin{w} L(w)$$

But as we've discussed, training error is not the only thing to care about. So, we could imagine modifying our quality metric to not just care about error, but to also factor in other notions such as the magnitude of the coefficients. Let's assume we have some function $R(w)$ to describe the magnitude of our coefficients (more on what this function will be in a bit). We can then modify our quality metric to balance these two functions. The $\lambda$ value is there to act as a weighting between these two notions of cost (error and magnitude); more on how to choose $\lambda$ later.

$$\hat{w} = \argmin{w} L(w) + \lambda R(w)$$

The main idea here now is when we are optimizing for coefficients, it's not just the lowest error coefficients, but the ones that minimize some combination of error and cost from having high magnitude. $\lambda$ is a weighting parameter between these two cost functions. When $\lambda$ is small, we care more about the error term $L(w)$ in our optimization. When $\lambda$ is large, we care more about the magnitude term $R(w)$ in our optimization. $\lambda$ is called a **regularization hyperparameter**.

Before discussing $\lambda$ in detail, let's discuss specific choices of this $R(w)$ function to measure magnitude.

## $R(w)$ Measuring Magnitude

```{margin}
{{ref_quality_metric}}\. Recall, the quality metric defines what is "best". A different quality metric will, by definition, change which predictor we consider is the "best" one.
```

This $R(w)$ function is supposed to take all of our coefficients and return a number summarizing their overall magnitude. We call this $R$ function the **regularizer** or **regularization cost**.

Just like almost everything else we will see in this book on machine learning, how we define the magnitude of our models coefficients is yet another *modeling choice*. Choosing a different measurement of magnitude will result in different properties of which learned predictor will be considered the "best"<sup>{{ref_quality_metric}}</sup>.

### Idea 1: Sum of Coefficients

The most intuitive idea that we can imagine to describe the magnitude of all of our coefficients is to simply just sum them up.

$$R(w) = \sum_{j=1}^D w_j$$

While this notion of a regularizer seems intuitive and straightforward, it won't quite work 😞. Because we are just adding up the coefficients, we will run into the problem of coefficients with differing signs "cancelling each other out". So in a model where $w_1 = 1,000,000,000$ and $w_2 = -1,000,000,000$, $R(w) = 0$ indicating the coefficients have small magnitude, when in fact, they are quite large. So while this is on the right track, it's not quite there.

### Idea 2: Sum of Absolute Coefficients

A simple modification to the last idea to fix the major problem with it is to take the absolute values of the coefficients instead.

$$R(w) = \sum_{j=1}^D \lvert w_j \rvert$$

```{margin}
{{ref_norm_soon}}\. More on this terminology and notation at the end of this section.
```
This now solves the problem with our first idea since there is no longer any "cancelling out". This regularizer is actually quite popular in practice and is often called the **LASSO regularizer** or regularizing with the **L1 Norm**<sup>{{ref_norm_soon}}</sup>. We will discuss this regularizer in detail in the next chapter (TODO link). This regularizer is popular enough that it gets its own notation

$$R_{lasso}(w) = \lVert w \rVert_1^1$$

### Idea 3: Sum of Squared Coefficients

Similar to the last idea, another way to avoid coefficients "cancelling out," is to square the coefficients instead of taking the absolute value

$$R(w) = \sum_{j=1}^D w_j^2$$

This regularizer is called the **Ridge regularizer** or regularizing with the **L2 Norm**. Like the last regularizer, it also has its own common notation

$$R_{ridge}(w) = \lVert w \rVert_2^2$$

We will spend the rest of this chapter discussing the Ridge regularizer and the impacts of regularization on our model.

```{admonition} Aside: p-norms
There is a common set of terminology and notation for measuring the magnitude of an array/vector of numbers called the **norm**. The norm is a generalized notion of magnitude of a collection of numbers, and a common set of these norms are called **p-norms**. A p-norm is defined as

$$\lVert w \rVert_p^p = \sum_{j=1}^D \lvert w_j \rvert^p$$

or written in the equivalent form

$$\lVert w \rVert_p = \sqrt[p]{\sum_{j=1}^D \lvert w_j \rvert^p$$

The intuition is to measure the "length" (magnitude) of a vector of numbers. Two of the most common p-norms are the L1 norm ($p=1$) and L2 norm ($p=2$). You can define p-norms for many different values of $p$, but these are by far the most common instances.
```

## Ridge Regression

TODO

## Old Stuff


```{video} ../../_static/regression/linear_regression/manim_animations/true_function_anim.mp4
```