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

# <i class="fas fa-book fa-fw"></i> Bias and Fairness

Now that we have seen concrete examples of models and learning algorithms in both the settings of regression and classification, let's turn back to the discussion we introduced at the beginning of the book about the use of ML systems in the real world. In the {doc}`../../intro/index`, we discussed a few examples of deployed ML models that can be biased in terms of the errors or the predictions they make. We highlighted the Gender Shades study that showed that facial recognition systems were more accurate on faces of lighter-skinned males than darker-skinned females. We also discussed the PredPol model that predicted crime happening way more frequently in poorer communities of color, simply by the choices of which crimes to feed the model in the first place and the feedback loops that creates.

You will have no trouble trying to find other examples of machine learning systems' biased outputs causing harm. From [Google's ad algorithm showing more prestigious jobs to men](https://www.washingtonpost.com/news/the-intersect/wp/2015/07/06/googles-algorithm-shows-prestigious-job-ads-to-men-but-not-to-women-heres-why-that-should-worry-you/), to [Amazon's recruiting tool that was biased against women](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight-idUSKCN1MK08G), to [the COMPAS model used in the criminal justice system falsely predicting Black individuals were more likely to commit crimes](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing).

Clearly it is possible to spot model's that behaviors are not ideal and the results in these headlines are clear signs of bias *somewhere* in our ML system. But the question is where does that bias come from? How does it effect our models? And importantly, what steps can we take to detect and hopefully prevent these biased outcomes from happening in the first place? In this chapter and the next, we will highlight some of the current research into the workings of ML models when the interact with society, and how we can more formally define concepts of bias and fairness to prevent our models from causing harm before we deploy them.

This chapter will tell this story about defining bias and fairness in three major parts:
* Background: The COMPAS Model
* Where does Bias come from?
* How to define fairness?

Before beginning this journey, it is important to note that my experience has being squarely from the field of machine learning and computer science. While many of these discussions of how to encode concepts of fairness into models are relatively new in our field (~15 years), these are not necessarily new concepts. Scholars of law, social justice, philosophy and more have all discussed these topics in depth long before we have, and we will only be successful by working with them in unison to decide how to build our models.

## Background: The COMPAS Model

A few years ago, a company called NorthPointe created a machine learning model named COMPAS to predict the likelihood that an arrested person would *recidivate*. Recidivation is when a person is let out of jail, where they eventually recommit that crime. Recidivating is an undesirable property of the criminal justice system, as it potentially means that criminals were being let go just to go repeat the crimes they were originally arrested for. So the modelers were hoping use data to better predict who was more likely to recidivate could help the criminal justice system. This model was deployed and was used by judges to help them determine details of sentencing and parole, based on how likely the person was predicted to be a risk of reoffending.

The modeling task COMPAS set out to solve was to take information about a person who was arrested, and predict how likely they would be to recidivate. The person who was arrested was given a survey asking questions about their life such as "Was one of your parents ever sent to jail or prison?" and "How many of your friends/acquaintances are taking drugs illegally?" Important to note for what will come up, the survey *never* asks the person about their race or ethnicity. The COMPAS model then takes this survey and computes a score from 0 to 10. This regression task indicates that a score near 0 is unlikely to recidivate and a score of 10 means they are very likely to recidivate. The training data for this model was based on survey results of past persons who were arrested and the labels to calibrate the scores were if that person was re-arrested in 2 years of being released.

[ProPublica](https://www.propublica.org/about/), an independent non-profit newsroom, did an [analysis in 2016](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) that claimed that this COMPAS model was biased against Black people who were arrested. ProPublica's piece outlines that the risk scores that COMPAS predicted for Black people were more likely to be in the "high risk" category (scores closer to 10) than White people with comparable backgrounds. That, along with other details, led them to the conclusion that this COMPAS model was biased against Black people.

```{margin}
{{ref_propublica_follow_up}}\. ProPublica followed up on this response with their own. Read it [here](https://www.propublica.org/article/technical-response-to-northpointe).
```

Northpointe, the model's creators, [countered in their own article](https://www.documentcloud.org/documents/2998391-ProPublica-Commentary-Final-070616.html) that the ProPublica analysis was flawed and that if you look at the data, the scores they predicted are well **calibrated**. Calibration is a measure for risk-score models like COMPAS to measure how likely their scores relate to the true probability of recidivating. So for example, a model is well calibrated if, when it predicts a risk score of 2/10, there is about a 20% chance that the person actually recidivates, as compared to when it predicts a score of 9/10, there is about a 90% chance that the person actually recidivates. So their claim is the that the model is in fact well calibrated, and their scores are accurate to reality<sup>{{ref_propublica_follow_up}}</sup>.

So this leaves us with some key questions: Who is right here? Do either of these claims make us feel more or less confident about using a model like this in the criminal justice system?

Well, in a sense, they are actually both correct. At least, you can use the same data to back up *both of their claims*.

Consider Northpointe's claim of the model being well calibrated. The following graph from the outputs of the model and comparing it to the rate of recidivism. The graph shows the calibration curve for the model separating its scores for Black/White people. Northpointe's claim is that these scores are close to the line $y = x$, meaning good calibration. And to bolster their point, the calibration is more-or-less the same across the groups Black/White. Clearly the lines aren't *exactly* the same, but the gray error regions show uncertainty due to the noise in the data, and those regions of uncertainty overlap by a lot indicating the two trends are reasonably close given the amount of certainty we have.

```{figure} calibration.png
---
alt: Recidivism rate by risk score and race. White and black defendants with the same risk score are roughly equally likely to reoffend. The gray bands show 95 percent confidence intervals.
width: 100%
align: center
name: northpointe_calibration
---
Recidivism rate by risk score and race. White and black defendants with the same risk score are roughly equally likely to reoffend. The gray bands show 95 percent confidence intervals. ([Source: The Washington Post](https://www.washingtonpost.com/news/monkey-cage/wp/2016/10/17/can-an-algorithm-be-racist-our-analysis-is-more-cautious-than-propublicas/))
```

But it turns out that the *same data* can also back up ProPublica's point! In particular, if you look at the rate of mistakes made by the model, it is more likely to label Black people who would ultimately not reoffend with a higher risk. Consider changing this problem from a regression one to a classification one, where the labels are "Low Risk" (negative) and "High Risk" (positive). In this setting, ProPublica is arguing that the *false positive rate* for Black people is much higher than the FPR for White people. This can be seen in the graph below, where a larger percentage of *Black People who did not reoffend* are disproportionately rated as higher risk.


```{figure} fpr.png
---
alt: Distribution of defendants across risk categories by race. Black defendants reoffended at a higher rate than whites, and accordingly, a higher proportion of black defendants are deemed medium or high risk. As a result, blacks who do not reoffend are also more likely to be classified higher risk than whites who do not reoffend.
width: 100%
align: center
name: propublica_fpr
---
Distribution of defendants across risk categories by race. Black defendants reoffended at a higher rate than whites, and accordingly, a higher proportion of black defendants are deemed medium or high risk. As a result, blacks who do not reoffend are also more likely to be classified higher risk than whites who do not reoffend. ([Source: The Washington Post](https://www.washingtonpost.com/news/monkey-cage/wp/2016/10/17/can-an-algorithm-be-racist-our-analysis-is-more-cautious-than-propublicas/))
```

In this graph, we can actually still see the truth of Northpointe's claim. Within each category (low/high), the proportion that reoffended are the same regardless of race. But the fact that a larger shade of "Did not reoffend" (dark blue) are labeled High Risk for Black people is Propublica's claim of bias.

So how can both actually be right? Well, as we will see later in this chapter, defining notions of fairness can be a bit tricky and subtle. As a bit of a spoiler for the discussion we will have over this chapter and next, there will never be one, right, algorithmic answer for defining fairness. Different definitions fairness satisfy different social norms that we may wish to uphold. It's a question of which norms (e.g., non-discrimination) we want to enshrine in our algorithms that is the important discussion to be had, and that requires people with stakes in these systems to help decide. Our book, can't possibly answer what those decisions should be in every situation, so instead, we are going to try to introduce the ideas of some possible definitions of bias and fairness and talk about how they can be applied in different situations. This way, you have a set of "fairness tools" under your toolbelt to apply to the situation based on what you and your stakeholders decide are important to uphold in your system.

## Bias

Before defining concepts of "what is fair", let's first turn to a discussion of where this bias might sneak in to our learning systems. Importantly, if we believe this COMPAS model is making biased outcomes, we should ask "Where does this bias come from?"

As we addressed in the {doc}`../../intro/index`, most times bias enters our model through the data we train it on (Garbage in $\rightarrow$ Garbage out. Bias in $\rightarrow$ Bias out). It probably wasn't the case that some rogue developer at Northpointe intentionally tried to make a racist model. Instead, it is more likely that they unintentionally encoded bias from the data the collected and fed to the model. However, if you remember carefully, we said earlier on in the description of the COMPAS model that it doesn't even use race/ethnicity as a feature! How could it learn to discriminate based on race if it doesn't have access to that data?

The simple answer is even though it didn't ask about race explicitly, many things it asks can be highly correlated to race. For example, due to historic laws that prevent Black families from purchasing property in certain areas in the past, continue trends today of certain demographics being overrepresented in some areas and underrepresented in areas. Zip code is a somewhat informative proxy of race. Now consider that with all of the other somewhat informative correlations you can find with race (health indicators, economic status, name, etc.) the model can inadvertently pick up the signal that is race from all of those proxy features.

The story of where bias enters our model is as bit more complex than just that though. It is worth diving in to the various points of our ML pipeline that can be entry points for bias to our model. Most of this section of the chapter is based on the incredible [A Framework for Understanding Unintended Consequences of Machine Learning (Suresh & Guttag, 2020)](https://arxiv.org/abs/1901.10002). In their paper, they outline six primary sources of bias in a machine learning pipeline that we will introduce and summarize. These sources are:

* **Historical bias**
* **Represenation bias**
* **Measurement bias**
* **Aggregation bias**
* **Evaluation bias**
* **Deployment bias**

```{figure} bias.png
---
alt: An ML pipeline with the sources of the six biases listed above
width: 100%
align: center
name: bias_pipeline
---
An ML pipeline with the 6 sources of bias entering at various points.
```

### Historical Bias

**Historical bias** is the reflection that data has on the real world it is drawn from. The world we live in is filled with biases for and against certain demographics. Some of these biases were ones that started a long time ago (e.g., slavery) that can have lasting impacts today, others are biases that emerge over time but can still be present and harmful (e.g., trans-panic). Even if the data is "accurately" drawn from the real world, that accuracy can reflect the historical and present injustices of the real world.

Some examples:

* Earlier we introduced the concept of redlining. A common practice historical, illegal today, still has lasting effects due to housing affordability and transfer of generational wealth.
* The violence against Black communities on the part of police make data in the criminal justice system related to race *extremely biased* against Black people.
* In 2018 only about 5% of Fortune 500 CEOs were women. When you search for "CEO" in Google Images, is it appropriate for Google to only show 1 in 20 pictures of women? Could reflecting the accurate depiction of the world perpetuate more harm?

Historical bias is one of the easiest forms of bias to understand because it draws from the concepts of bias we understand as human beings. That doesn't mean it is always easy to recognize though. A huge source of biases come from *unconscious* biases that are not necessarily easy for us to spot when they are in our own blind-spots.

### Representation Bias

**Representation bias** happens when the data we collect is not equally representative of all groups that should be represented in the true distribution. Recall that in our standard learning setup, we need our training to be a representative sample of some underlying true distribution that our model will be used in. If the training data is not representative, then our learning is mostly useless.

Some examples of representation bias in machine learning include:

* Consider using smartphone GPS data to make predictions about some phenomena (e.g., how many people live in a city). In this case, our collected dataset now only contains people who own smartphones with GPSs. This would then underestimate counts for communities that are less likely to have smartphones such as older and poorer populations.
* [ImageNet](https://www.image-net.org/) is a very popular benchmark dataset used in computer vision applications with 1.2 million images. While this dataset is large, it is not necessarily representative of all settings in which computer vision systems will be deployed. For example, about 45% of all images in this dataset were taken in the US. A majority of the remaining photos were taken in the rest of North America and Western Europe. Only about 1% and 2.1% of the images come from China and India respectively. Do you think a model trained on this data would be accurate on images taken from China or India (or anywhere else in the world) if it has barely seen any pictures from there?

### Measurement Bias

**Measurement bias** comes from the data we gather often only being a (noisy) proxy of the things we actually care about. This one can be subtle, but is extremely important due to how pervasiive it is. We provide many examples for this one to highlight its common themes.

* For assessing if we should give someone a loan, what we care about is whether or not they will repay that loan through some notion of financial responsibility. However, that is some abstract quantity we can't measure, so we use someone's Credit Score as a hopefully useful metric to measure something about financial responsibility.
* In general we can't predict the rate that certain crimes happen. We don't have an all-mighty observer who can tell us every time a certain crime happens. Instead, we have to use arrest data as a proxy for that crime. But note, that arrest rate is just a proxy for crime rate. There are many instances of that crime that may not lead to arrests, and people might be arrested for not that crime at all!
* For college admissions, we might care about some notion of intelligence for admissions to college. But intelligence, like financial responsibility, is some abstract quality not a number. We can try to come up with tests to measure *something* about intelligence, such as IQ tests or SAT scores and use those instead. But note those a proxies of what we want, not the real thing. And worse, those tests have been shown to not be equally accurate for all subgroups of our population; IQ tests are notoriously biased towards wealthier, whiter populations. It's likely that these measure have *nothing* to do with the abstract quality we care about.
* If factory workers are predicted that they need to be monitored more often, more errors will be spotted. This can result in **feedback loops** to encourage more monitoring in the future.
    * This is the *exact* thing we saw with PredPol system in the introduction, where more petty crimes fed into the model led to more policing, led to more petty crimes being fed to the model.
* Women are more likely to be misdiagnosed (or not diagnosed) for conditions where self-reported pain is a symptom since doctors frequently discount women's self-reported pain levels. In this case "Diagnosed with X" is a biased proxy of "Has condition X."

So measurement bias is bias that enters our models from what we choose to measure and how, and often that the things we measure are (noisy) proxies of what we care about or not even related at all!

In particular, we have actually tricked you with a form of measurement bias throughout this whole chapter to highlight how tricky it can be to spot. Recall when we introduced the COMPAS we said the goal was to predict whether or not someone would recidivate. Again, recidivating is recommitting the crime they were arrested for sometime in the future. How did Northpointe gather their training data for make their model? They looked at past arrests and found whether or not that person was arrested again within two years. Think carefully: **Is what they are measuring the same as the thing they want to measure?**

No. They are not the same. The data "arrest within two years" is a *proxy* for recidivating. Consider how they differ:

* Someone can re-commit a crime but not be arrested.
* Someone can be arrested within two years, but not for the same crime they were originally arrested for.
* Someone can re-commit a crime more than two years after their original arrest.

Now consider all of the ways that those differences can affect different populations inconsistently. With our PredPol (predictive policing) example earlier, and the discussion of over-policing of Black communities. It is completely unsurprising that there is a higher arrest rate among Black people who previously were arrested, even if it wasn't some recidivating (e.g., maybe they were originally arrested for robbery, but then re-arrested for something petty like loitering). This is one of the reasons why when you look at the COMPAS data, you see a higher "recidivism" rate for Black people. Because they aren't actually measure recidivsim, they are measuring arrests!

Measurement bias sneaks in to ML systems in *so many* sneaky ways when model builders don't think critically about what data they are including and why and what they are actually trying to model. A general rule of thumb is that anything interesting you would want to model about a person's behavior rely on some, if not all, measures of proxy values.

### Aggregation Bias

Often times we try to find a "one size fits all" model that can be used universally. **Aggregation bias** occurs when using just one model fails to serve all groups being aggregated over equally.

Some examples of aggregation bias:

* HbA1c levels are used to monitor and diagnose diabetes. It turns out HbA1c levels differ in complex ways across ethnicity and sex. One model to make predictions about HbA1c levels across all ethnicities and sexes may bne less accurate than using separate models, even if everyone was represented in the training data equally in the first place!
* General natural language processing models tend to fail on accurate analysis of text that comes from sub-populations with a high amount of slang or hyper-specific meanings for more general words. As a lame, Boomer example, "dope" (at one point) could have been recognized by a general language model as meaning "a drug", when younger generations usually say it to mean "cool" by a general language model as meaning "a drug", when younger generations usually say it to mean "cool".

### Evaluation Bias

**Evaluation bias** is similar to representation bias, but is more focused on how we evaluate and test our models and how we report those evalutions. If the evaluation dataset or the benchmark we are using doesn't represent the world well, we can have an evaluation bias. A **benchmark** is a common dataset used to evaluate models from different researchers (such as ImageNet).

Some example of evaluation bias:

* Similarly, the ImageNet example would also suffer from evaluation bias since its evaluation data is not representative of the real wrold.
* It is common to just report a total accuracy or error on some benchmark dataset and using a high number as a sign of a good model without thinking critically about the errors it makes.
    * We saw earlier in the Gender Shades example that there was drastically worse facial recognition performance when used on faces of darker-skinned females. The common benchmarks for these tests that everyone was comparing themselves on only had 5-7% faces of darker-skinned women.

### Deployment Bias

**Deployment bias** can happen when there is a difference between how a model was intended to be used, and how the model is actually used in real life.

Some example of deployment bias include:

* Consider the context of predicting risk scores for recidivism rates. While it may be true that the model they developed was calibrated well (its scores were proportional to rate of recidivism), their model wasn't designed or evaluated in the context of a judge using the scores to decide parole. Even if its calibrated in aggregate, how do they know how it performs on individual cases where a human then takes its output and then makes a decision. How the model was designed and evaluated didn't match how it was used.
    * People are complex, and often don't understand the intricacies of modeling choices made by the modelers. They might make incorrect assumptions about what a model says and then make incorrect decisions.

### Recap

So now with a brief definition of all of these biases, it would be good to look back at this ML pipeline Suresh & Guttag introduced to get a full picture of how they enter and propagate throughout and ML system ({numref}`bias_pipeline`).

```{figure} bias.png
---
alt: An ML pipeline with the sources of the six biases listed above
width: 100%
align: center
---
```

## Fairness

Now that we have discussed sources of bias in our model, we have a clearer idea of precisely how they can enter our ML pipeline. While the outputs of the model being biased feels unfair, it turns out that we will not get fairness for free unless we can demand it from our models. What we are interested in exploring now is how to come up with mathematical definitions of fairness, such that we can compute some number(s) to tell us if biases in our model are ultimately affecting its predictions in an undesired way.

```{margin}
{{ref_facct}}\. See the ACM Conference on [Fairness Accountability, and Transparency (FAccT)](https://facctconference.org/)
```

There is a lot of active research <sup>{{ref_facct}}</sup> on how to come up with mathematical definitions of fairness that we can enforce and verify. The hope is that by coming up with some mathematical definition, we can hopefully spot early in the process when the model might exhibit unfair outputs.

In this chapter, we will introduce a few different possible definitions of fairness, and in the next chapter we will look at comparing them to see if they can be cohesive. In particular in this chapter, we will be interested in coming up with definitions for a concept known as **group fairness**. Group fairness, also called **non-discrimination** is the intuitive concept that fairness means your membership of some group based on some uncontrollable property of yourself (race, gender, age, nationality, etc.) shouldn't negatively impact the treatment you receive. Group fairness notions are ones that try to prevent discrimination such as racism, sexism, ageism, etc.

For the remained of this chapter, we are going to focus on a cartoonishly simplified example of college admissions to try to reveal the key intuitions of how we can potentially define fairness. Note that the assumptions in this example are clearly not correct, but we use it for its simplicity and to act as a proof of concept. The intent is that by showing behaviors and properties of an overly simplified, contrived scenario, we can better understand what we can expect in real life. All of the things we will introduce in this chapter and next can be generalized to the complexities of the real world.

```{important} College Admissions - Credit
This example is borrowed from the fantastic [The Ethical Algorithm](https://duckduckgo.com/?q=the+ethical+algorithm+1g&ia=web) by [Michael Kearns](https://www.cis.upenn.edu/~mkearns/) and [Aaron Roth](https://www.cis.upenn.edu/~aaroth/). This book is an incredible overview of the current work done to embed socially-aware values, such as fairness, into algorithms. We include their example scenario to introduce concepts of fairness because it is that good. Our contribution to the example is adding mathematical definitions and notation to make the ideas they present in this book for general audiences more suitable for a technical one.
```

So in this example, we are in charge of building a college admissions model. This model is supposed to take information about a college applicant, and predict whether or not we should let them in. In this scenario, we are going to make the following (unrealistic) assumptions:

* The only thing we will use as an input to our model is the students' SAT score. Clearly real college admissions systems use more information and include holistic reviews, but we will limit this for simplicity.
* For every applicant, there is some single measurable notion of "success" and the goal of our model is to let in students who will meet this definition of success. Depending on the priorities of your college, you could imagine defining success as 1) will the student graduate, or 2) will the student graduate and get a job or 3) will the student graduate and give a ton of money back to the school. Depending on how you define success, that defines which examples are considered "positive" and "negative" examples. Clearly "success in college" is more complex than just one goal, as every individual student might have their own notion of what being successful in college means. For simplicity though, we will assume there is a consistent and universal notion of "successful college student", whatever that may be.
* There is no notion of competition or limited spots. If everyone would truly meet the definition success, then everyone would be let in. But that doesn't mean we want to have our model let everyone in since if they ultimately wouldn't be successful, that could be a huge waste of time and money for us and the student.
* ```{figure} circle_square.png
  ---
  alt: A pie chart showing the 66% demographics of Squares and 33% circles
  width: 60%
  figclass: margin
  align: center
  ---
  ```

  To talk about demographics and concepts of group fairness, we will assume all of our applicants are part of one of two races: Circles and Squares. In this world, Circles make up a majority of the population (66%) and Circles (33%). Consider that in this world Circles also face systematic oppression, and often are economically disadvantaged due to the barriers they face in their lives.

Now again, all of this is an extreme over simplification of the real world, but it will still be useful for us to get some key intuitions for what fairness might mean in the real world.

### Notation

Before defining fairness formally, we need to introduce some notation to describe the situation we defined above. In our ML system, our training data will be historical data of students and if they met our definition of success. Recall that the only input we will use in this example is the SAT score, but we will also track the demographics of Circle/Square to understand concepts of fairness. The notation will be general for more complex scenarios, and we highlight what values they take in our example

```{admonition} Notation
$X$: Input data about a person to use for prediction

* Our Example $X = \text{SAT Score}$

$A$: Variable indicating which group $X$ belongs to

* Our Example: $A = \square$ or $A = \bigcirc$

$Y$: The true label

* Our Example: $Y = +1$ if truly successful in college, $Y = -1$ if not

$\hat{Y} = \hat{f}(X)$: Our prediction of $Y$ using learned model $\hat{f}$

* Our Example: $\hat{Y} = +1$ if predicted successful in college, $\hat{Y} = -1$ if not
```

### Fairness Definitions

```{margin}
{{ref_gerrymandering}}\. We see a similar problem in coming up with a rigorous notion of [Gerrymandering](https://en.wikipedia.org/wiki/Gerrymandering) in politics. When you see examples, it's sometimes easy to point out that something looks off. But coming up with a formal, working definition to work in all cases is challenging and is a choice of which priorities to include in your definition. See l
```

In our setup, we might be concerned that our college admissions example may potentially be biased against Circles given the fact that they are a smaller portion of the population, and we may be concerned about the adverse affects of the discrimination they face in the world affecting our college admissions choices. But how can we definitively know what discrimination is or if our system is unfair? This is where we introduce the concept of coming up with measure to describe how fair a system is, and if we see that fairness is being violated, is a clear indicator that our system is discriminatory <sup>{{ref_gerrymandering}}</sup>. We'll see different

In the following sections, let's explore some mathematical definitions of concepts of fairness using the notation we outlined above. Recall that we are focused on notions of group fairness, where we don't want someone's outcome to be negatively impact by membership in some protected group (in this case their race being Circle or Square).

We also show for each section a brief code example to show how to compute the numbers in question. The code examples assume you have a `DataFrame` `data` with the all of the data and predictions.

| X   | A   | Y   | Y_hat |
|-----|-----|-----|-------|
| ... | ... | ... | ...   |

#### "Shape Blind"

One of the most intuitive notions to define fairness is the simple idea to make your process completely blind to the concept you may worry it can discriminate on. So in our example, the hope is by simply withholding the race of the applicant from our model, then discrimination is not possible since the model couldn't even use that as an input to discriminate on. This approach is often called **"fairness through unawareness."**

While intuitive, this just doesn't work in practice. Think back to our COMPAS example. The model was able to have discriminatory outcomes *without* using race as an input. That means in the COMPAS example, even though it was "color blind", its decisions were anything but.

One reason this approach doesn't work in the real world is that there are often subtle correlations between other features not related to the attribute you may wish to protect (e.g., zip code and race). Even if we leave that feature out of our model, it can still inadvertently learn it from other features.

You may think we could just remove any correlated features, but that would mean removing almost any data to use at all. For example, SAT scores are also correlated with race even if the Circles/Squares would be equally successful in college. One factor that impacts SAT scores is how much money you have to afford SAT prep. If the Squares are generally richer and can afford SAT prep, their scores are artificially higher even if they aren't necessarily more successful in college! So if we wanted to remove any features with correlation, then we couldn't even use our one feature SAT score!

#### Statistical Parity

So since being completely unaware of race to achieve fairness wasn't feasible, let's explore notions that try to compare how the model behaves for each of the groups of our protected attribute $A$. The simplest notion to compare the behavior on groups is known as **statistical parity**. Statistical parity dictates that in order for a model to be considered fair, the predictions of the model need to satisfy the following property.

$$P(\hat{Y} = +1 | A=\square) \approx P(\hat{Y} = +1 | A=\bigcirc)$$

In English, this says that the rate that Squares are admitted by our system should be approximately equal to the rate that the rate that Circles are admitted. So for example, if the admitted rate for Squares is 45%, then the admitted rate for Circles should be 45% as well. Note that we use approximately here and in other definitions because demanding *exactly equal* probabilities is often hard. Instead we usually define a range of acceptable distance such as 0.1% and won't call it biased if the Square admission rate is 45% and the Circles is 45.1%.

Let's consider some tradeoffs of using this definition to define if our admissions model is fair.

**Pros**

* The definition is simple, and easy to explain. This is a positive because now users can better understand what we are measuring and when this definition of fairness is violated.
* It only depends on the model's predictions, so can be computed live as the model makes new predictions.
* This definition aligns with certain legal definitions of equity so there is established precedent in it being used.

**Cons**
* In some sense, it is a rather weak notion of fairness because it says nothing about how we accept these applicants or if they are actually successful. For example, a random classifier satisfies this definition of fairness even though it completely disregards notions of success. There are other, more nefarious, models that could be used that still meet statistical parity such as a classifier that is accurate for Square applicants, and then just chooses the same percentage of non-successful Circle applicant. It doesn't seem fair that you can set up the Circles to fail even though they are equally represented.
  * Do note that that the fact that a random classifier can satisfy this definition fairness is actually not all that much of a critique. In fact, it is a proof of concept that it is even possible to satisfy this definition of fairness. There are some useful notions of fairness which might not have algorithms that can actually satisfy them!
* A more serious objection is that, in some settings, the rates for our labels might not be consistent across groups. Statistical parity is making the statement that the rates that which groups meet this positive label (collegiate success in our example), is consistent across groups. To be clear, it is very possible that the rates of success across groups is actually the same, but it's possible that it's not and forcing statistical parity in that case might not be right.

Let's consider a different example to make that last point clearer. Consider a model that predicts if someone has breast cancer, and a positive label is the detection of breast cancer. Would we say the breast cancer model is discriminatory because $P(\hat{Y} = +1 | A = \text{woman}) = 1/8$ while $P(\hat{Y} = +1 | A = \text{man}) = 1/833$? No, because clearly the base-rate of the phenomena we are predicting is different between men and woman. Forcing statistical parity in this model to consider it fair would mean either incorrectly telling more women they don't have breast cancer (when they might actually) or telling more men that they do have breast cancer (when they in fact do not). In the case where the base rates differ, demanding statistical parity makes no sense.

Back our college admissions example, the same critique could apply if we assume the base rates of success are different. However, many people argue that in the college setting, the base rates for success should be equal enough (i.e., race does not effect your success in college), so statistical parity is appropriate. We will see in our next chapter how this assumption is a statement of a particular worldview about how fairness should operate.

The following code example shows how to compute the numbers we care about for statistical parity.

```python
squares = data[data["A"] == "Square"]
circles = data[data["A"] == "Circle"]
square_admit_rate = (squares["Y_hat"] == +1).sum() / len(squares)
circle_admit_rate = (circles["Y_hat"] == +1).sum() / len(circles)

assert math.isclose(square_admit_rate, circle_admit_rate, abs_tol=0.01)
```


#### Equal Opportunity

To account for the critiques of statistical parity ignoring the actual success of the applicants, we arrive at a stronger fairness definition called **equal opportunity** or the **equality of false negative rates (FNR)**.

Equal opportunity is defined as the following. Recall that a false negative and true positive are the only possibilities when we are considering the true label are positive.

$$P(\hat{Y} = +1 | A=\square, Y=+1) \approx P(\hat{Y} = +1 | A=\bigcirc, Y=+1)$$

Note that this definition is almost the same as statistical parity, but adds the requirement that we only care about consistent treatment of Squares/Circles that were ultimately successful. In English, this is saying the rate that successful Squares and successful Circles are admitted should be approximately equal.

The intuition here is to remember our discussion on {ref}`the relative costs of false negatives and false positives <classification:confusion_matrix>`. If in our setting of college admissions, we assume the cost of a false negative (denying someone who would have ultimately been successful) is worse, then a reasonable definition of fairness would demand that the rate of those false negatives should be approximately equal across subgroups. Note that the definition above is about true positive rate (TPR) which is related to FNR by $TPR + FNR = 1$.

As its own definition of fairness, equal opportunity has its own set of pros/cons in its use.

**Pros**

* Much better since it controls from the true outcome. Fixes all of the cons of statistical parity.

**Cons**

* This definition only controls for equality in terms of the positive outcomes. It doesn't make any care about fairness in terms of the negative outcomes.
* It is more complex to explain to non-experts. With even a little complexity, challenges can occur since its important everyone understands and agrees on what is at stake to be considered fair.
* It requires actually knowing the true labels. This is fine to check if your model is unfair on the training data, but it is not possible to actually measure equal opportunity on new examples if you don't know their true labels.

```python
success_squares = data[(data["A"] == "Square") & (data["Y"] == +1)]
success_circles = data[(data["A"] == "Circle") & (data["Y"] == +1)]

# Note TPR + FNR = 1
tpr_squares = (success_squares["Y_hat"] == +1).sum() / len(success_squares)
tpr_circles = (success_circles["Y_hat"] == +1).sum() / len(success_circles)

assert math.isclose(tpr_squares, tpr_circles, abs_tol=0.01)
```

#### Predictive Equality

Just like equal opportunity, **predictive equality** or the **equality of false positive rate (FPR)** takes the same form but for controlling the rate for negative examples. All of the commentary is the same for predictive equality but which numbers you compute are different. Useful when you care more about controlling equality in the negative outcome.

$$P(\hat{Y} = -1 | A=\square, Y=-1) \approx P(\hat{Y} = -1 | A=\bigcirc, Y=-1)$$

```python
no_success_squares = data[(data["A"] == "Square") & (data["Y"] == -1)]
no_success_circles = data[(data["A"] == "Circle") & (data["Y"] == -1)]

# Note TNR + FPR = 1
tnr_squares = (no_success_squares["Y_hat"] == -1).sum() / len(no_success_squares)
tnr_circles = (no_success_circles["Y_hat"] == -1).sum() / len(no_success_circles)

assert math.isclose(tnr_squares, tnr_circles, abs_tol=0.01)
```

### Which Metric Should I Use?

Here we've just outlined 3, but there are [many many more](https://fairmlbook.org/) definitions you can consider.

So which one should you use? Or more specifically, considering our COMPAS example which one would we use to measure as a determination of the models predictions are fair or not?

Unfortunately, we cannot tell you in general. The reason is each definition makes its own statement on what fairness even means. Choosing a fairness metric is an explicit statement of values that we hold when thinking about fairness. Many of these values are important values, but are fundamentally assumptions about how we believe the world works. These assumptions are often unverifiable. In our next chapter, we will explore contrasting worldviews and how they impact our understanding of fairness.

### How do I Use These Metrics?

That is a simpler question to answer. A simple approach is to use them as part of your model evaluation. When comparing models of different types and complexities, you can use the metric of your choice to report a measure of each model's fairness on both the training set and some validation set. Depending on your context, you might demand a certain threshold for your fairness metric before deploying your model.

There are also techniques to augment our standard learning algorithms to make them fairness-aware in the first place. Some algorithms are more complex than others, but they simply involve changing how we measure quality to care about fairness instead of just minimizing error. One simple approach is to use regularization, but instead of penalizing large coefficients, penalize coefficients that lead to larger disparities in some fairness metric.

## Recap

Discrimination in ML models is a crucial problem we need to work on. It has real impacts, on real people, and its happening right now whether or not we are aware of it.

In general, this will not be a problem that will be solved algorithmically. We can use algorithms to help, but choosing which algorithms to use in the first place It’s not a problem that will only be solved algorithmically. We need people (e.g., policymakers, regulators, philosophers, developers) to be in the loop to determine the values we want to encode into our systems and which values we want to uphold.

In the next chapter, we will explore some limitations of how we have defined fairness so far, and think more critically about the worldviews we assume when defining fairness ideas.
