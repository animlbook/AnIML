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