import pandas as pd
from collections import Counter

df = pd.read_csv("datasets/email_spam_ham.csv")

df_dict = df.to_dict(orient="list")

test_dict = {
    "texts": [
        "free money now claim prize",
        "win free cash prize today",
        "meeting scheduled for tomorrow",
        "please review the attached report",
    ],
    "labels": [1,1,0,0]
}

class NaiveBayesClassification:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.classes = []
        self.classes_count = {}
        self.classes_word_count = {}
        self.classes_total_words = {}
        self.all_words_count = Counter()

    def display(self):
        print(self.classes)
        print(self.classes_count)
        print(self.classes_word_count)
        print(self.classes_total_words)
        print(self.all_words_count)

    # create a dictionary with word count
    def fit(self, texts: list, labels: list):
        self.classes = sorted(set(labels))

        for c in self.classes:
            self.classes_count[c] = labels.count(c)
            self.classes_word_count[c] = Counter()
            self.classes_total_words[c] = 0

        for text in texts:
            self.all_words_count.update(text.split())

        for text, label in zip(texts, labels):
            text_counter = Counter(text.split())
            self.classes_word_count[label].update({word: text_counter.get(word, 0) for word in self.all_words_count})

        for c in self.classes:
            self.classes_total_words[c] = self.classes_word_count[c].total()

    def predict_one(self, text):
        prior_probabilities = {}
        class_results = {}

        messages_count = sum(self.classes_count.values())
        
        for c in self.classes:
            prior_probabilities[c] = self.classes_count[c] / messages_count

            product = prior_probabilities[c]
            for word in text.split():
                product *= ((self.classes_word_count[c][word] + self.alpha) / (self.classes_total_words[c] + (len(self.classes_word_count[c]) * self.alpha)))

            class_results[c] = product

        return max(class_results, key=class_results.get)



nb = NaiveBayesClassification()
nb.fit(test_dict["texts"], test_dict["labels"])
print(nb.predict_one("please review the attached report for the prize money"))
            


        



