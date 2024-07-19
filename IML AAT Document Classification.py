import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read training data
with open("trainingdata.txt", "r") as f:
    lines = f.readlines()

num_train_samples = int(lines[0].strip())
train_data = [line.strip() for line in lines[1:num_train_samples + 1]]

categories = []
documents = []

for line in train_data:
    parts = line.split()
    categories.append(parts[0])
    documents.append(" ".join(parts[1:]))

# Read test data from standard input
input = sys.stdin.read
test_lines = input().strip().split("\n")
num_test_samples = int(test_lines[0].strip())
test_documents = [line.strip() for line in test_lines[1:num_test_samples + 1]]

# Vectorize the documents using CountVectorizer
vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X_train = vectorizer.fit_transform(documents)
y_train = categories
X_test = vectorizer.transform(test_documents)

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict categories for test documents
predicted_categories = classifier.predict(X_test)

# Output the predictions
for category in predicted_categories:
    print(category)
