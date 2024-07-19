# IML_AAT
Introduction to Machine Learning

## Document Classification
[Link to Document Classification](https://www.hackerrank.com/challenges/unbounded-knapsack/submissions/code/391221785)

```python
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


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


input = sys.stdin.read
test_lines = input().strip().split("\n")
num_test_samples = int(test_lines[0].strip())
test_documents = [line.strip() for line in test_lines[1:num_test_samples + 1]]


vectorizer = CountVectorizer(lowercase=True, stop_words='english')
X_train = vectorizer.fit_transform(documents)
y_train = categories
X_test = vectorizer.transform(test_documents)


classifier = MultinomialNB()
classifier.fit(X_train, y_train)


predicted_categories = classifier.predict(X_test)


for category in predicted_categories:
    print(category)

```

## Day 3: Basic Probability Puzzles #6
[Link to Day 3: Basic Probability Puzzles #6 submission](https://www.hackerrank.com/challenges/basic-probability-puzzles-6/submissions/code/392185696)

```python
rom fractions import Fraction

def calculate_probability(W_A, B_A, W_B, B_B):

    P_white_A = Fraction(W_A, W_A + B_A)
    P_black_A = Fraction(B_A, W_A + B_A)
    
    P_black_after_white = Fraction(B_B, W_B + 1 + B_B)
    P_black_after_black = Fraction(B_B + 1, W_B + B_B + 1)
    
    P_black_B = P_white_A * P_black_after_white + P_black_A * P_black_after_black
    return P_black_B

W_A = 5
B_A = 4
W_B = 7
B_B = 6

result = calculate_probability(W_A, B_A, W_B, B_B)

print(f"{result.numerator}/{result.denominator}")


```

## Stack Exchange Question Classifier
[Link to Stack Exchange Question Classifier submission](https://www.hackerrank.com/challenges/stack-exchange-question-classifier/submissions/code/392182517)

```python
import json,sys
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
if sys.version_info[0]>=3: raw_input=input
transformer=HashingVectorizer(stop_words='english')

_train=[]
train_label=[]
f=open('training.json')
for i in range(int(f.readline())):
    h=json.loads(f.readline())
    _train.append(h['question']+"\r\n"+h['excerpt'])
    train_label.append(h['topic'])
f.close()
train = transformer.fit_transform(_train)
svm=LinearSVC()
svm.fit(train,train_label)

_test=[]
for i in range(int(raw_input())):
    h=json.loads(raw_input())
    _test.append(h['question']+"\r\n"+h['excerpt'])
test = transformer.transform(_test)
test_label=svm.predict(test)
for e in test_label: print(e)

```


## Laptop Battery Life
[Link to Laptop Battery Life submission](https://www.hackerrank.com/challenges/battery/submissions/code/392182219)

```python
import pandas as pd
from sklearn.linear_model import LinearRegression


timeCharged = float(input())
data = pd.read_csv('trainingdata.txt', names=['charged', 'lasted'])
train = data[data['lasted'] < 8]
model = LinearRegression()
model.fit(train['charged'].values.reshape(-1, 1), train['lasted'].values.reshape(-1, 1))
ans = model.predict([[timeCharged]])
print(min(ans[0][0], 8))
```

## Stock Predictions
[Link to Stock Predictions submission](https://www.hackerrank.com/challenges/stockprediction/submissions/game/392181698)

```python
from __future__ import division
from math import sqrt
from operator import add
from heapq import heappush, heappop

def printTransactions(money, k, d, name, owned, prices):
    def mean(nums):
        return sum(nums) / len(nums)

    def sd(nums):
        average = mean(nums)
        return sqrt(sum([(x - average) ** 2 for x in nums]) / len(nums))

    def info(price):
        cc, sigma, acc = 0, 0.0, 0
        for i in range(1, 5): 
            if price[i] > price[i - 1]: cc += 1
        sigma = sd(price)
        mu = mean(price)
        c1, c2, c3 = mean(price[0:3]), mean(price[1:4]), mean(price[2:5])

        return (price[-1] - price[-2]) / price[-2]
    
    infos = list(map(info, prices))
    res = []
    
    drop = []
    
    for i in range(k):
        cur_info = info(prices[i])
        if cur_info > 0 and owned[i] > 0:
            res.append((name[i], 'SELL', str(owned[i])))
        elif cur_info < 0:
            heappush(drop, (cur_info, i, name[i]))
    
    while money > 0.0 and drop:
        rate, idx, n = heappop(drop)
        amount = int(money / prices[idx][-1])
        if amount > 0:
            res.append((n, 'BUY', str(amount)))
            money -= amount * prices[idx][-1]
    
    print(len(res))
    for r in res:
        print(' '.join(r))

if __name__ == '__main__':
    m, k, d = [float(i) for i in input().strip().split()]
    k = int(k)
    d = int(d)
    names = []
    owned = []
    prices = []
    for data in range(k):
        temp = input().strip().split()
        names.append(temp[0])
        owned.append(int(temp[1]))
        prices.append([float(i) for i in temp[2:7]])

    printTransactions(m, k, d, names, owned, prices)

```
