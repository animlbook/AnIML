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
    ref_propublica_follow_up: 1
    ref_product: 2
    ref_facct: 3
    ref_gerrymandering: 4
---

```{code-cell} ipython3
:tags: [remove-input]

# If you are reading this notebook on a Binder, make sure to right-click the file name in
# the left-side file viewer and select `Open With > Notebook` to view this as a notebook
```

# <i class="fas fa-book fa-fw"></i> Decision Trees

In this chapter, we will introduce a popular type of model called a **Decision Tree**. Decision trees are based on the intuitive decision-making tool humans often use called a flow chart. For example, here is the University of Washington's flowchart that they share with students, staff, and faculty to help them determine which actions to take when they contract or are exposed to COVID. A flowchart (or decision tree) asks as sequence of questions, and you follow the path depending on how you answer the questions. Ultimately the path leads to a node that gives you a decision that you can act on.

```{figure} uw_covid.png
---
alt: A flowchart describing what to do when exposed to COVID or test positive
width: 100%
align: center
---
University of Washington's Public Health Flowchart for COVID-19 Pandemic (2023)
```

## Parametric vs Non-parametric Methods

So as we've mentioned before, we will be learning a plethora of different models in the remainder of this book. As we suggested, it helps to compare/contrast models to better understand their properties and why you may want to use one over the other.

One way to group models into what assumptions they make about the world and how they are formulated. One distinction that is commonly made is between **parameteric models** and **non-parametric models**.

**Parametric models** are ones that make an assumption about the distribution of the data and require learning some finite number of parameters. Linear regression, Logistic Regression, and Naïve Bayes are all examples of parametric models. Linear regression, for example, assumes there is a linear relationship between the inputs and outputs, and there is some normally-distributed noise around the the linear function. To learn the model, we learn a number of parameters $\hat{w} \in \mathbb{R}^D$ for our $D$-dimensional features $h(x)$. A useful approximation for parametric models are ones where you can write down the predictions as a tidy formula such as $\hat{y} = \hat{w}^Th(x)$.

On the other hand **non-parametric models** are ones that (mostly) don't make strong assumptions about the data distribution, and/or aren't represented by a fixed number of parameters. This is often described as models that can scale in complexity based on how much training data is available. In the rest of this chapter, we will discuss our first non-parametric model with Decision Trees.


## Loan Safety

For this chapter and the next, we will be switching our example scenario from predicting sentiment of a review, to predicting whether or not it is safe to give a loan to a potential applicant. In terms of a modelling task, we will gather information from the applicant about their credit history, their income, how long of a term they are looking for the loan, and other information about them. We will use that information as inputs for our model to lead to predictions of Safe (+1) or Risky (-1). Below, we show an (made up) example dataset for this task.

```{table} Example Loan Safety Dataset
:name: loan_safety

| Credit    | Term  | Income | Label  |
|-----------|-------|--------|--------|
| excellent | 3 yrs | high   | safe   |
| fair      | 5 yrs | low    | risky  |
| fair      | 3 yrs | high   | safe   |
| poor      | 5 yrs | high   | risky  |
| excellent | 3 yrs | low    | safe   |
| fair      | 5 yrs | low    | safe   |
| poor      | 3 yrs | high   | risky  |
| poor      | 6 yrs | low    | safe   |
| fair      | 3 yrs | high   | safe   |
```

Before discussing the ML algorithm we will discuss and how we make predictions, it's important to make sure we are asking questions about what we are modeling and why. Consider our discussions from the last chapter on {doc}`../bias_fairness/index`. What concerns might we have about potential biases in our data or fairness concerns we should consider for our predictions? We asked our students this question, and they outlined just a couple of many concerns or questions we should explore in depth before deploying such a model.

* What are the effects of the errors are model will make? Should we be concerned about disparate cost between a false-positive and false-negative? Can there be a compounding effect of using our model that might lead to a financial crisis like the one we went through in 2008? How do we audit our model to ensure it doesn't cause those effects?
* There are likely biases present in our training data, which will cause biased outcomes of our predictions. Just for example, there are many economic factors that we think about affecting loan safety that may disproportionately make a group less qualified when they are actually just as loan-worthy. For example, black Americans have suffered from disproportionate economic hardships due to factors such as redlining, inequitable access to high-paying jobs, and any other reason we can imagine structural bias affecting information about loan safety.
* What legal constraints does are model need to abide by? Is there a requirement on how fairness should be defined? Even if there aren't legal requirements, which fairness values to we care to uphold in our model?

We won't provide answers to these questions in this chapter, not because they aren't important, but because they are fundamentally outside the realm of algorithmic solutions. Just like in {doc}`../bias_fairness/index`, these are questions that need to be answered, but require human discussion and input for what social values we want to encode in our model and why. In this chapter, we will largely skip over these questions to focus on the core concepts of the modeling task. Everything we discuss in this chapter can be augmented with the fairness concepts we discussed earlier.

## Decision Trees

At a high level, a **Decision Tree** is a flowchart about our data to make predictions about the labels. We ask questions, going left or right down the tree based on the answers, to come to a prediction. The nodes in the middle of the tree are called **branch nodes** or **internal nodes** that ask questions about the input data while the decisions (Safe/Risky) are stored in the **leaf nodes** at the bottom. You might find that terminology a bit backwards, but you can imagine it as an upside-down tree where the root is at the top and the leaves are at the bottom.


```{code-cell} ipython3
---
tags:
  - remove-input
  - remove-output
mystnb:
  image:
    width: 60%
    align: center
---
# TODO This doesn't work well for most of our diagrams so I'm hiding it entirely. Would love to come back and make these
# look nice.

from graphviz import Digraph

BRANCH_STYLE = {"shape": "diamond", "fillcolor": "#F2F4FF", "fontcolor": "#474973", "style": "filled"}
SAFE_STYLE = {"label": "Shape", "shape": "oval", "fillcolor": "#4B8F8C", "fontcolor": "#FFFFFF", "style": "filled"}
RISKY_STYLE = {"label": "Risky", "shape": "oval", "fillcolor": "#BB7E8C", "fontcolor": "#FFFFFF", "style": "filled"}

SAFE_FONT = "#6E8031"
RISKY_FONT = "#963334"

def font_color(text, color):
  return f'<FONT COLOR="{color}">{text}</FONT>'

# Create Digraph object
dot = Digraph()

# Make nodes and edges
dot.node("Start", shape="box")

dot.node("Credit?",  **BRANCH_STYLE)
dot.node("safe1", **SAFE_STYLE)
dot.node("term1", label="Term?",  **BRANCH_STYLE)
dot.node("Income?",  **BRANCH_STYLE)
dot.edge("Start", "Credit?")
dot.edge("Credit?", "safe1", xlabel="excellent")
dot.edge("Credit?", "term1", label=" fair")
dot.edge("Credit?", "Income?", label="poor")

dot.node("risky1", **RISKY_STYLE)
dot.node("safe2", **SAFE_STYLE)
dot.edge("term1", "risky1", xlabel="3 yrs")
dot.edge("term1", "safe2", label=" 5 yrs")

dot.node("term2", label="Term?",  **BRANCH_STYLE)
dot.node("risky2", **RISKY_STYLE)
dot.edge("Income?", "term2", xlabel="high")
dot.edge("Income?", "risky2", label="low")

dot.node("risky3", **RISKY_STYLE)
dot.node("safe3", **SAFE_STYLE)
dot.edge("term2", "risky3", xlabel="3 yrs")
dot.edge("term2", "safe3", label=" 5 yrs")

dot
```

figure

Let's explore how to use this tree by finding the prediction for the sixth example in {numref}`loan_safety` with fair credit, a 5 year term, and low income. The tree makes the following steps starting from the top.

* What is the credit of the applicant? Their credit is fair, so we go down the middle branch.
* What is the term limit of the loan? Their term limit is 5 years, so we go down the right branch.
* We are at a leaf node marked Safe, so we predict Safe.

The decision tree itself is quite intuitive for making predictions since it really does mirror a flowchart. The challenge though is *learning the best decision tree from the data itself*. In the following sections, we will discuss the algorithm we will use for learning our tree.

##  Visualizing Trees



