import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

classifier = skflow.LinearClassifier(n_classes=3)

print("Training the classifier...")
classifier.fit(iris.data, iris.target)
print("Evaluating the classifier...")
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
