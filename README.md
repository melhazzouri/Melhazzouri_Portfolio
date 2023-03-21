# Mahmoud El Hazzouri Portfolio
Data Science Portfolio

# Project 1: DL Chatbot- DoorDash Customer Support: Project Overview
In the past few years, chatbots have become wildly popular in the tech and business sectors. These intelligent bots are so adept at imitating natural human languages and conversing with humans, that companies across various industrial sectors are adopting them. From e-commerce firms to healthcare institutions, everyone seems to be leveraging this nifty tool to drive business benefits. In this project, a Customer Support Chatbot was built and deployed. The Artificial Intelligent Chatbot was trained using Deep Learning algorithms and deployed using the Flask method.

#### Keys: NLP Chatbot - RNN - Flask
![image](https://user-images.githubusercontent.com/39967400/226722294-99e0d9b5-5b31-4ff2-82df-68b1969f86ee.png)

# Project 2: French To English Translation: Project Overview
Within the European Union, French holds the distinction of being the fourth most widely spoken mother tongue, and depending on who you ask, it may even be the second most prevalent.A staggering 80 million individuals worldwide claim French as their first language, with 61 million of them residing in France. French-speaking communities can be found dispersed throughout various regions of the world.

Despite the language's global popularity as the "language of love," it remains challenging for many individuals to acquire French fluency. This is a hurdle that I, too, have encountered, having recently relocated to Montreal, Quebec.  I am using this project to build a Neural Machine Translation model to translate from French to English. 

#### Keys: NLP - NMt - RNN
![image](https://user-images.githubusercontent.com/39967400/226723042-9711606c-38b4-4691-a1d4-d22c1aa7971e.png)

# Project 3: Sentiment Analysis Using LSTM
RNNs have been applied with success to many kinds of natural language processing projects in the past few years. Those applications include speech recognition, language modeling, translation, and, of course, sentiment analysis. None of this would be possible without a special kind of RNN, and that is the LSTM (Long short-term memory). For many purposes, LSTMs work better than traditional recurrent networks.LSTM models are widely used nowadays, as they are particularly designed to have a long-term “memory” that is capable of understanding the overall context better than other neural networks affected by the long-term dependency problem. In this project, LSTM was used to carry out sentiment analysis over a set of Amazon reviews. 

#### Keys:  NLP Sentiment Analysis  - LSTM
![image](https://user-images.githubusercontent.com/39967400/226724309-1fc3bfb6-b5e9-44dc-bd53-8912f9b33112.png)


# Project 4: Amazon Reviews Sentiment Analysis
Sentiment Analysis is a prevalent method of text classification that assesses an incoming message and determines whether the sentiment expressed is positive, negative, or neutral. In today's world, where customers are more vocal about their thoughts and emotions, it is crucial for businesses to have a grasp on these sentiments. It can be challenging for a person to manually examine each statement and determine the underlying emotion. However, with the advancement of technology, companies are now able to automatically analyze customer feedback, be it from surveys or social media interactions, to gain insights into their customers' needs and preferences. This allows them to tailor their products and services accordingly. In this project, sentiment analysis was carried out over a set of Amazon reviews. SpaCy was used for tokenization, Lemmatization and Removing stopwords and scikit-learn to build the machine learning models for different batches of data: Logistic Regression, Decision Tree,  K-Nearest Neighbors Algorithm (KNN), Support Vector Classifier (SVC) and Naive Bayes. 

#### Keys: NLP Sentiment Analysis - Logistic Regression - Decision Tree- K-Nearest Neighbors Algorithm (KNN)- Support Vector Classifier (SVC)- Naive Bayes. 
![image](https://user-images.githubusercontent.com/39967400/226724026-c36e7f0b-c412-421e-9e4b-72763bd7119e.png)



# Project 5:  NLP - Amazon Reviews Clustering: Project Overview
Amazon is the largest E-commerce platform in the world. And, customer reviews are proven sales drivers, and something the majority of customers will definitely want to see before deciding to make a purchase. In this project, an NLP technique has been used to process a set of Amazon reviews, given by customers. Customer reviews were scrapped using BeautifulSoup. An NLP technique was used to process the reviews. The text has been preprocessed by removing numbers, punctuation, emojis, urls, and stopwords. The resulting text has been represented with Tf-Idf vectorization. The vector representation of the text has been used to perform text clustering. Clusters built with K Means algorithm, optimal number of clusters were found using the Elbow method, extraction of clusters topics through cluster centroids coordinates and visualization. It was possible to identify 5 different clusters that describe the segmentation of the customer reviews.

#### Keys: NLP - Clustering 
![image](https://user-images.githubusercontent.com/39967400/226723451-3585df8f-de86-4690-a435-e142e090bd4b.png)

# Project 6: Arabic Text Classification Using Deep Learning
Despite the significant progress made in Natural Language Processing, the majority of research endeavors have concentrated solely on the English language, while other languages remain relatively unexplored. Arabic, in particular, presents an imposing linguistic challenge due to its intricate grammar, vast lexicon, and diverse writing styles.Against this backdrop, this project is dedicated to the classification of Arabic text through the application of deep learning methodologies. Assembled as a comprehensive learning process, this project strives to comprehend the utilization of deep learning for text classification in ArabicThe model built yielded 75% accuracy. For future work, I plan to use pre-trained models as well as compare CNN and RNN for Arabic text classification.

#### Keys: NLP Text Classification - CNN

![image](https://user-images.githubusercontent.com/39967400/226724850-27a97801-9c93-4d19-bde8-4d5bf484232a.png)

# Project 7: NMT - Arabic to English Translation

Machine translation is a specialized area within computational linguistics that is dedicated to the automated conversion of text in one language to another language. In this process, the input text is already composed of symbols from a particular language, and the machine translation program must convert these symbols into symbols that correspond to another language.
Neural machine translation (NMT) is a machine translation approach that involves using an artificial neural network to predict the probability of a sequence of words. This method typically employs a single integrated model to generate translations of entire sentences. Due to the capabilities of neural networks, NMT has become the most powerful algorithm for machine translation. This cutting-edge algorithm utilizes deep learning techniques, where large datasets of translated sentences are utilized to train a model that can effectively translate between any two languages.To date, limited research has been conducted in the area of Arabic language processing. Here, a Recurrent Neural network (RNN) was built to to translate Arabic text into English using Keras. 

#### Keys: NLP - NMT - RNN
![image](https://user-images.githubusercontent.com/39967400/226725313-32fa2b2f-404e-4288-a370-4b895fe75b6a.png)


# Project 8: Arabic Font Classification 
In recent times, the identification of Arabic fonts has become increasingly significant, primarily because of its wide-ranging applications in various fields. The primary focus of this study centers around the recognition aspect of Arabic fonts, which poses significant challenges, notably the vast diversity present within the fonts utilized in the Arabic language.  A Convolutional neural network (CNN) was built to classify between Arabic fonts with 95% accuracy. 

#### Keys: Image Classification - CNN
![image](https://user-images.githubusercontent.com/39967400/226725660-7c64d0f9-4645-46fd-8a98-d923207e93e9.png)


# Project 9: CNN Image Classification for Bird's Species
In this project, we developed a deep learning convolutional neural network (CNN) classifier of bird species using Python 3, Keras, and Tensorflow. The model has been trained while different hyperparameters are tuned for best accuracy in predicting the bird species.

#### Keys: Image Classification - CNN
![image](https://user-images.githubusercontent.com/39967400/226726029-fbd5b425-7429-4dab-ad05-625e8552c09a.png)


# Project 10: Health Insurance Cost Estimation
Health insurance can get pricey. Understanding what helps determine your health insurance costs can help you save money, although not everything an insurance company considers is necessarily something you can control. This is an analysis of which factors determine the health insurance cost using Machine Learning. Four different models were developed for the health insurance cost estimation: Linear Regression, Decision Tree Regressor, Support Vector Regressor, and Random Forest Regressor. The model with the highest accuracy for health insurance cost estimation was deployed using Flask

#### Keys: Linear Regression - Decision Tree Regressor - Support Vector Regressor -  Random Forest Regressor - Flask

![image](https://user-images.githubusercontent.com/39967400/226726502-3b8f8ffd-9fa4-46b4-8755-8374c85aceba.png)

# Project 11: Customer’s Churn Predictions
There are various methods for measuring the churn rate, including the number of customers lost, the percentage of customers lost in comparison to the company's total customer base, the value of recurring business lost, or the percentage of recurring value lost. However, in this particular dataset, it is defined as a binary variable for each customer, and determining the rate is not the focus. The idea of the churn rate implies that there are factors that affect it, so the objective is to identify and quantify those factors. In this project, several machine learning models were tested: Logistic Classifier, Support Vector Classifier, Decision Tree Classifier, K-nearest Classifier and Random Forest Classifier as well as an Artificial Neural Network (ANN). Performance of all the models were compared. Logistic Regression resulted in the best performance of 81% accuracy

#### Keys:   Logistic Classifier - Support Vector Classifier - Decision Tree Classifier- K-nearest Classifier - Random Forest Classifier - ANN
![image](https://user-images.githubusercontent.com/39967400/226726782-1397f292-a35a-4459-96e6-2eede65d3b67.png)

# Project 12:  Regression Discontinuity Design - COVID
The goal of this project is to use RDD to estimate the effects of lockdowns and reopenings on the number of daily Covid cases. 
1. Extracted all the indices that contained the relevant dates needed.
2. Assigned date as our independent variable (x) and daily cases as our dependent variable (y).
3. Plotted the data in a scatter plot.
4. Assigned a cutoff predictor.
5. Added a new column named "day" as cardinal numbers to make it easier to plot and perform a regression on.
6. Transformed the data and performed a Regression Discontinuity.
7. Plotted the Final Regression Discontinuity Model.

#### Keys: RDD

![image](https://user-images.githubusercontent.com/39967400/226727603-49bd8ab8-ee22-4c3a-9582-d386cda17aca.png)


