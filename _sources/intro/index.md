# <i class="fas fa-hand-sparkles fa-fw"></i> Introduction

```{note}
This book is still under construction! There are many features, improvements we would like to make in the long term. Namely, some of the diagrams are currently hand-drawn and the hope is to improve these with animations and cleaner, code-generated figures in the future!

We are also always looking for feedback and help on the book. Feel free to join our GitHub community to help us improve the book!
```

Welcome! 🎉 !!!

If you're here, you've probably at least heard the term "machine learning" or "artificial intelligence" somewhere on the internet, the news, or somewhere in your day-to-day life. For better or for worse, machine learning has been an incredible force in shaping the modern world. You would be hard pressed to use any piece of modern technology (including any website) without the influence of machine learning powering some (most?) of your interactions.

Before outlining what this book is and how it's different than other books, let's look at an example to try to get a better understanding of what this "machine learning" thing means.

## Machine Learning in the Wild

We'll come back and formally define what machine learning is and the techniques to implement it. For now, let's just use the colloquial definition that *machine learning* is the tools and techniques that let a computer make predictions about the future given past data. This is a good enough approximation for us to begin to appreciate where this is used in our day-to-day tools.

Consider Netflix, the popular streaming platform for movies and TV shows. Just looking at their home page, where might you imagine all of the different places they might be using machine learning to power their website? Take a minute and try to list off places you think it might show up behind the scenes before expanding the box below.

![A screenshot of the Netflix home page showing the menu, a banner ad for the show "Never Have I Ever" and a list of recommended TV shows](netflix_home.png)

```{admonition} Where might there be Machine Learning in Netflix?
:class: tip, dropdown

Here are just a few of the ones we can think of at the top of our head, and almost all of these have to do with **recommendation**. Netflix uses computers to decide, based on your past viewing history, which shows it should recommend to make you more likely to watch the shows they recommend. Here are just a few places they shape their recommendations to you:

* Which show to show in the banner position (in the screenshot about it is "Never Have I Ever")
* Which shows to list in the categories (e.g., "TV Shows") below the banner
* Which categories to show you in the first place. If you like action movies, it will likely show you an action category.
* Even once they decide to recommend a show (e.g., "Stranger Things"), they also use machine learning to decide which cover art they should use that would make you more likely to watch that show.

  ![Various album arts in different styles for the show "Stranger Things"](netflix_stranger_things.png)

And there are so many more that we might not even know about! There are likely even algorithms behind the scenes helping Netflix decide which shows to invest in and even create in the future!
```


## Why Learning Machine Learning Matters

And all of this is just for one site used for watching TV! If there is so much going on behind the scenes there, imagine all of the ways it can be used in life-changing applications used in our society.

Machine Learning has amazing power to make applications from ones that are fun ([teaching a computer to make pasta](https://www.youtube.com/watch?v=Y_NvR5dIaOY)) to ones that can be used for great social good ([detecting ear infections with a smartphone app](https://www.youtube.com/watch?v=Y_NvR5dIaOY)).

But as the maxim of every superhero movie says, "with great power comes great responsibility." These machine learning tools are also used in incredibly high-stakes applications such as the criminal justice system, [where they have been found to produce biased outcomes](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing). And what's more, they are being used increasingly in more and more domains such as our elections, national defense, health, and transportation.

Without being informed about what these models are doing and how they work, we have no hope of being able to assess and critique where we should and *should not* be using them. Our hope in this book is that we can show you the awesome potential of these systems and how you to build them well. At the same time, we want you to be considering the serious ramifications of automation that these models afford and how to avoid some dangerous pitfalls that can end up harming people.

The remaining part of this chapter is broken into 3 parts. Before we give a broad overview of what ML is and the things you will see in this book at the end of this chapter, we we want to establish some of the values we are bringing in as authors of this book and how to effectively learn from this book.


## What is this Book?

### Our Values

Our values determine what we prioritize in structure and content. This is not to say these are the only things that should be valued, but these our the ones we use as a guiding principle for this book.

First, **it is critical that everyone understands how machine learning is used in our society**. We've already highlighted why we think everyone learning this is important, but wanted to write it down as a core value.

We intentionally used a pretty vague word there: "understand". To understand something can mean different things to different people depending on their goals and where they are in their educational journey. We can't determine the things you will get out of this book (we'll talk in a bit about how learning works), but the examples and applications we show try to hit a wide range of possibilities to give you ideas of what these systems can do and how they work internally.

Since there are so many concepts in ML and places they can be applied, we can't feasibly show them all. If we leave out a concept or application, it's not because it isn't important! Instead, it's almost always a matter of trying to use an example that's easier for a broader audience to understand.

```{margin}
1\. See Carol Dweck's *growth mindsets*
```

Second, **anyone can learn machine learning!** While this is true for any topic<sup>1</sup>, it's important to highlight in the case of machine learning. ML is notoriously seen as a difficult topic to master due to a traditional focus on advanced mathematical details and proofs. While those ideas are important for an expert in machine learning to know, they aren't necessary for forming an intuition for the key ideas of ML. So this book is not going to go rigorously into the mathematics behind ML, but instead tries to make the ideas intuitive and show you how to apply them.

```{margin}
2\. We do rely on a base-level understanding of probability and a willingness to learn new notation and concepts.
```
This does not mean we won't use math at all! While we try to minimize pre-requisite knowledge, we still require some<sup>2</sup>.  Mathematical notation can be very effective for communication and we use it frequently to communicate more precisely/effectively.

These two values hopefully give you a good idea of our approach for how we teach machine learning to a broad audience. If this does not sound like the book that meets what you are hoping to learn (e.g., you want a much more rigorous mathematical discussion of ML or proofs behind the techniques), we recommend some other books that are fantastic in those spaces.

* [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/), Trevor Hastie, Robert Tibshirani, Jerome Friedman.
* [Machine Learning: A Probabilistic Perspective](https://www.cs.ubc.ca/~murphyk/MLbook/), Kevin Murphy
* [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/), Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar

### How Learning Works

```{margin}
3\. This book is fantastic! I wish I read it while I was in college!
```

Learning is a complex process that whole fields of research focus on learning how learning works. The research of effective learning was summarized very well in [Make it Stick](https://www.retrievalpractice.org/make-it-stick) by Peter C. Brown, Henry L. Roediger III, and Mark A. McDaniel<sup>3</sup>.

The book outlines two very popular, but **ineffective**, learning strategies:

* *Re-reading* the same text multiple times until you "get it".
* *Massed practice*, or practicing the same skill/concept over and over until you get it right. This is a tendency many people have when cramming for something like a test.

Both of these strategies are popular because they make you *feel* like you are making progress. However, there is little evidence that they are actually effective strategies for longer-term retention or transfer of knowledge.

It turns out that learning effectively is hard work and will challenge you throughout the process; often if a strategy feels easy, that might be a good sign it's not effective. In "Make it Stick", they argue the most important thing to learning effectively is being an active participant in the process. When you are taking charge of your learning, you will face many setbacks; you shouldn't perceive those as failures, but instead as areas to grow. "Make it Stick" argues that active learning involves three key steps:

* **Practice retrieving new learning from memory**. As you are reading, constantly pause and reflect on the new things you have read. Quiz yourself on the new content to make sure you understood it the first time around. Spend time **generating** knowledge by trying to answer questions on your own before looking at the answer. **Elaborate** on the concepts you are learning as if you were describing them to someone else; this forces you to summarize what you learned and synthesize it with your past knowledge.
* **Spaced repetition is incredibly important**. Learning a concept isn't a "one-then-done" ordeal. Effectively using what you learned requires retrieving information from memory. One of the best ways to practice this retrieval is to increasingly space out the times in which you test retrieving the information (e.g., 1 day, 1 week, 1 month, etc.).

  A great tool for managing this spacing is [Anki](https://apps.ankiweb.net/). Anki is a flashcard app that has a built-in features for spaced-repetition; it gives you a challenge to master a certain number of cards a day and spaces out when you see certain cards based on that mastery.

* ```{margin}
  4\. See [this study](https://pubmed.ncbi.nlm.nih.gov/8084699/) about the effects of mixed-practice on professional baseball players.
  ```

  **Interleaving your practice**. Instead of practicing all the concepts until mastery for Chapter 1 before moving to Chapter 2, you should mix it up and test them together. Retrieving knowledge in real life rarely has a context like "I'm studying for Chapter 1 right now, so I only need to think of Chapter 1 things". In reality, you will need to practice bringing up any subset of knowledge. Therefore, your practice should reflect that goal if you want to see improvements of retrieval in realistic contexts<sup>4</sup>.

When learning about ML in our book, we recommend that you employ these three study tips. You should be building up meaningful notes, reflecting what you've learned and relating it to past concepts, testing yourself regularly, and spacing-out and mixing-up your practice. We highly recommend (would require it if we could) using a tool like Anki to help you test yourself on concepts. It's rare that we care about a word-for-word definition of a term or formula, but instead care about the intuition and understanding that comes from being familiar with that concept. Don't penalize yourself when studying for messing up a single word as long as you can assess that you have understood that concept (don't worry, it will always come back later in Anki).


## Introduction to Machine Learning

One of the most general, albeit pretty abstract, definitions of **machine learning** is one of the most cited ones:

```{epigraph}
A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P** if its performance at tasks in **T**, as measured by **P**, improves with experience **E**.

-- Tom M. Mitchell
```

```{figure} ml-steps.png
---
scale: 25%
figclass: margin
alt: A visual pipeline showing "Data" to "ML Method" to "Intelligence"
---
Pictorial representation of these steps.
```

This is just trying to state in a very formal manner that machine learning is a program that improves at some task by learning from data rather than a human hard-coding the rules of the task.

For example, consider the task of building a system that classifies an image as containing a dog or a cat. One approach would be to try to write a program (e.g., a Python program) that looks at the pixels of the picture in a big `if/else` statement to make the decision. You (nor I) would really have any idea where to start writing such a program.

```{margin}
5\. We will call this the *training data* in future sections.
```

Instead, the machine learning approach is to feed a bunch of images that have already been labelled as cat images or dog images as input to some learning algorithm<sup>5</sup>, and letting that algorithm learn the patterns from the data itself. The ultimate goal of such a system is to train the computer to not just be able to answer this dog-vs.-cat question for images it has seen, but also to generalize to any dog or cat picture it might encounter in the future.

We'll come back to a more detailed explanation of this pipeline in the next chapter.

## Overview of Book

### Case Studies

We will tackle this introduction to machine learning with 5 overarching case studies to guide us. As we mentioned before, the contexts that these case studies focus on are not the only applications that are of import in machine learning. We spend the majority of our focus on these five examples for consistency and simplicity. This helps you form more explicit connections between the concepts. Each case study will span more than one chapter as indicated in the table of contents; each chapter corresponds roughly to one lecture of the UW CSE/STAT 416 course.

The five case studies we will examine in this book are:

* **Regression**: Learning how to predict the price of a house.
* **Classification**: Learning how to identify if a restaurant review is good or bad (sentiment analysis).
* **Document Retrieval and Clustering**: Given a news article, suggest similar ones that the reader might be interested in.
* **Recommender Systems**: Given the past purchases of a user, what products should we recommend?
* **Deep Learning**: Recognizing objects in images.

### Topics

As a brief (and incomplete) list of topics we will cover:

* **Models**: Different machine learning models that can be used.
  * Linear regression, regularized approaches (ridge, LASSO)
  * Linear classifiers: logistic regression
  * Non-linear models: decision trees and ensemble methods
  * Nearest neighbors, clustering, kernel methods
  * Recommender systems
  * Neural networks and deep learning
* **Algorithms**: Different algorithms for learning these models.
  * Gradient descent and its variations
  * Boosting and bagging
  * K-means
  * Agglomerative and divisive clustering
  * Hyperparameter optimization
* **Concepts**: High-level ideas from machine learning, computer science, statistics, and mathematics.
  * Point estimation, Maximum likelihood estimation (MLE)
  * Loss functions, bias-variance tradeoff, cross-validation
  * Sparsity, overfitting, model selection
  * Decision boundaries

With that, we have introduced what this book is about and some intentions for how it is structured. Our hope is that you find this book exciting and helps you build intuition for this incredibly important field!