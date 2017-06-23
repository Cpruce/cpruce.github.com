---
layout: post
title:  "Triage Prediction, for Fun and Profit"
date:   2017-06-19 13:38:00 -0700
categories: Machine Learning Data Science Bug Classification SVM 
---


During my time with a previous employer, the issue of assigning team-level responsibility, or *triaging*, was a central role in the engineering workflow, particularly among certain teams. Unfortunately and fortunately, my team was one of those lucky bunch which received a likely disproportionate share of PR's incoming, yet ending elsewhere. I was even luckier in stumbling across an awesome opportunity. 

![Try not to get bogged down in bugs](/img/triage_post/bug.png)

Each team handles triaging differently. In my team, we did round-robin on a monthly-basis. Having seen first-hand the [lack of scale](http://ijesc.org/upload/de10a74ab9db0ac338a3466fc6574001.Auto%20Bug%20Triage%20a%20Need%20of%20Software%20Industry.pdf) to this process, heard of the request to automate the process, and being very into Data Science and Machine Learning, I felt I was the right one for the task.

Right off-the-bat, I saw that I would structure the problem as a supervised, NLP problem. The ground truth would rest in the categories assigned to *closed*, *fixed* problem reports. The idea behind narrowing the set considered is to limit noise, ie. open PR's are currently unsettled and mistaken can act as a red herring. 

Though I knew the problem was pervasive at large companies, I thought quantifying the data to show the potential impact would be best for the trajectory of the project. An anonymized version of my data analysis is available [here](https://www.github.com/Cpruce/GNATS_Triage_Prediction/blob/master/triage_analysis.ipynb). This analysis scopes out and highlights a typical use case of triaging in a large company, as well as the potential gains an automation of process would provide. If you use GNATS as a bug tracking system, congratulations! Maybe this will be of help. Don't be afraid to reach out if so.

![](/img/triage_post/anon_plots.png)

Before entering into the spot checking phase, I was pretty sure that the process would take time, on the order of months. I had my LinearSVC with 5 recommendations hitting >95% accuracy, precision, and recall after a week or two. The main takeaway should be don't get discouraged. The library and community support today, especially with Python, is outstanding and it is easy to go from 0 to 60, real quick.

>"Human beings are constantly learning and teaching with every social interaction, even with himself/herself."

Under the gun, I used a sledgehammer in feature engineering: simply stripping all special characters and using the synopsis only gave the best results. Due to the significant amount of parsing, I decided to leave the audit-trail alone. Further supporting this intuition, the description field turned out to be very noisy, which makes sense as the description is 9 times out of 10 significantly less succinct. That said, if extracted and selected properly, there lie valuable information that can be built upon. Bigrams did give another boost, leaving trigrams with little room for improvement. Using feature extraction algorithms like PCA and [LSA](http://ieeexplore.ieee.org/abstract/document/5298419/citations?tabFilter=papers) to remove correlation/redundancy definitely should be explored too. I'd also tried [stemming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.6144&rep=rep1&type=pdf) with little or no visible gain.

Check out the [pipeline]() on Github.

Trying a few pre-packaged feature extractors/selectors from scikit-learn, I found that using the Term Frequency-Inverse Document Frequency Vectorizer (`TfidfVectorizer`) and removing English stop words compiled in the sklearn module gave pretty good results. 

Other models such as Neural Networks and KNN should be explored too. Incorporating those two into the bag of tools would help maximize the breadth of types of models used. This will help gauge Occam's Razor as well: if two solutions are equally performant, pick the simpler one. This can be mathematically formulated as <em>min(complexity) + min(error) = max(simplicity) + max(correctness)</em>. This doesn't need to always be explicitly stated, but it is good to consider.

The ML [models](https://www.researchgate.net/publication/308417176_Automated_Bug_Triaging_in_an_Industrial_Context) that I ended up comparing were `MultinomialNB`, `RandomForestClassifier`, `LogisticRegression`, and `LinearSVC`.

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Throughout my analysis and construction of the pipeline, there were tradeoffs I needed to consider, many of which I have not touched upon in the post and writeups. To name a few:

- **Number of Recommendations**: In order to maximize model robustness without diminishing user experience, I felt 5 recommendations was a good number to optimize this objective. This intuition was supported by the [Top-k Multiclass SVM](http://www.ml.uni-saarland.de/Publications/LapHeiSch-TopkMulticlassSVM2015.pdf) paper. Analysis of this decision will be shown by user interaction.
- **New/old Features**: New features that haven’t been seen before are essentially as good as random guessing. For convenience, I started with batch learning. However, OnlineSVM can be used to improve everyday, or however granular one would like.
- **Quality of Data**: Not all synopses are descriptive. Worse, not all data is reliable. Additionally, the Description field is noisy and yields little or no error reduction. There is certainly good information there. Two next steps are proper feature engineering of the Description field in order to extract the relevant information as well as filtering/correcting labels.
- **Depth of Problems**: Some classifications are just inherently difficult. If the data is small, doesn’t contain characteristic features, and/or describes a symptom that could have multiple origins, the classifier may become uncertain. This is part of the machine learning problems understanding. There is bound to be some error, which is desired to be minimized. 
- **Level of Automation**: I chose to give recommendations, in order to give way to an engineer confident in the category. I also got the bright idea to take a page out of search engines' books and thought it would just be a cool way to get better results as well =).
- **Algorithms/Techniques Decision**: Given the data I had, the problem at hand, and the tools at my disposal, I looked for the combination that gave the most satisfactory results within a short amount of time. Other approaches such PCA, Neural Networks, and LSA should certainly be explored in order to maximize effectiveness.



