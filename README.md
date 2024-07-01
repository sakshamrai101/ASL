# American Sign Language Recognition 

The problem we are aiming to solve is the classification of American Sign Language Characters to provide an ability of communication from deaf people to those who do not understand ASL. More specifically, we are looking to compare and contrast the efficacy between Supervised and Unsupervised classification machine learning algorithms. For the purpose of this project, we have decided 2 unsupervised clustering algorithms: Gaussian Mixture Model and K-Means clustering and 2 supervised models: Computational Neural Networks and Support Vector Machine algorithms. These models are picked because of their implementation and their common application in image recognition tasks, which aligns with the nature of the ASL dataset.

With the 27,455 available training data samples and 7172 samples allocated for testing, we are looking to measure the success of each model with different evaluation metrics (specific to each classification model). In addition to this, we also test the accuracy of our clusters by inputting test cases from making the signs on our webcams and observing if the results share a similar accuracy with the test data. This can be replicated as with the publicly available data - which in our case comes from the Google Open Teachable Machine, which is used as our benchmark model.
