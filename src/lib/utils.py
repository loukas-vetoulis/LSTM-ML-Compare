import sys
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from collections import Counter
import numpy as np
from scipy.stats import entropy

def downloadData():
    (x_train, y_train), (x_test, y_test) = imdb.load_data()

    # Combine and split data
    x_combined = np.concatenate([x_train, x_test])
    y_combined = np.concatenate([y_train, y_test])

    # Split the data 75/25
    x_train, x_test, y_train, y_test = train_test_split(x_combined, y_combined, test_size=0.25, random_state=1)
    return x_train,x_test,y_train,y_test

def text_to_vector(review, vocabulary_set):
    """Convert a review (list of word indices) to a binary vector"""
    return np.array([1 if word_id in review else 0 for word_id in vocabulary_set])

def vocab_cutter(n, k,x_train):
    # Count word frequencies in training data
    word_counts = Counter()
    for review in x_train:
        unique_words = set(review)  # if the world apears on the text
        word_counts.update(unique_words)
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Remove n most frequent and k least frequent words
    filtered_words = sorted_words[n:-k]

    # Create vocabulary set
    vocabulary = {word_id for word_id, _ in filtered_words}
    return vocabulary

# Calculate information gain
def information_gain_calculation(vocabulary,x_train,y_train):
    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    H_Y = entropy([num_pos/len(y_train), num_neg/len(y_train)], base=2)

    # Calculate information gain for each word
    info_gain = {}
    for word_id in vocabulary:
        # Get reviews containing this word
        containing_reviews = [i for i, review in enumerate(x_train) if word_id in review]
        
        if containing_reviews:  # if word appears in any review
            # Get labels for reviews containing this word
            labels_with_word = [y_train[i] for i in containing_reviews]
            pos_count = sum(labels_with_word)
            neg_count = len(labels_with_word) - pos_count
            
            # Get labels for reviews not containing this word
            labels_without_word = [y_train[i] for i in range(len(y_train)) 
                                if i not in containing_reviews]
            pos_count_without = sum(labels_without_word)
            neg_count_without = len(labels_without_word) - pos_count_without
            
            # Calculate conditional entropy
            total_reviews = len(y_train)
            p_word = len(containing_reviews) / total_reviews
            
            if pos_count + neg_count > 0:
                H_Y_given_w_true = entropy([pos_count/(pos_count + neg_count), 
                                        neg_count/(pos_count + neg_count)], 
                                        base=2) if pos_count + neg_count > 0 else 0
            
            if pos_count_without + neg_count_without > 0:
                H_Y_given_w_false = entropy([pos_count_without/(pos_count_without + neg_count_without),
                                        neg_count_without/(pos_count_without + neg_count_without)],
                                        base=2) if pos_count_without + neg_count_without > 0 else 0
            
            H_Y_given_w = p_word * H_Y_given_w_true + (1 - p_word) * H_Y_given_w_false
            info_gain[word_id] = H_Y - H_Y_given_w
    return info_gain

# Select top m words by information gain
def information_gain_cutting(info_gain,m,x_train,x_test):
    selected_words = sorted(info_gain.items(), key=lambda x: x[1], reverse=True)[:m]
    final_vocabulary = sorted({word_id for word_id, _ in selected_words})

    # Convert reviews to binary vectors
    X_train_binary = np.array([text_to_vector(review, final_vocabulary) for review in x_train])
    X_test_binary = np.array([text_to_vector(review, final_vocabulary) for review in x_test])

    print("Training set size:", X_train_binary.shape)
    print("Test set size:", X_test_binary.shape)

    return X_train_binary, X_test_binary
