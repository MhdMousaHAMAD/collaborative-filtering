from math import sqrt
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import csv
import time


# Return Pearson Correlation Coefficient for rows at key1 and key2 in dataset dictionary
# Rows represent users or items when the dictionary is transposed
def similarity_pearson(dataset, key1, key2):
    # Get mutual items
    mutual_items = {}
    for item in dataset[key1]:
        if item in dataset[key2]:
            mutual_items[item] = 1

    # If there are no ratings in common, return 0
    if len(mutual_items) == 0:
        return 0

    # Sum of the ratings
    sum1 = sum([dataset[key1][item] for item in mutual_items])
    sum2 = sum([dataset[key2][item] for item in mutual_items])

    # Sum of the rating squares
    sum1_squares = sum([pow(dataset[key1][item], 2) for item in mutual_items])
    sum2_squares = sum([pow(dataset[key2][item], 2) for item in mutual_items])

    # Sum of the products
    sum_product = sum([dataset[key1][item] * dataset[key2][item] for item in mutual_items])

    # Calculate r (Pearson score)
    numerator = sum_product - (sum1 * sum2 / len(mutual_items))
    denominator = sqrt(
        (sum1_squares - pow(sum1, 2) / len(mutual_items)) * (sum2_squares - pow(sum2, 2) / len(mutual_items)))
    if denominator == 0:
        return 0

    score = numerator / denominator
    # Normalize score to be between 0 and 1
    score = (score - (-1)) / (1 - (-1))
    return score


# Return Cosine Similarity for items at key1 and key2 in dataset dictionary
def similarity_cosine(dataset, key1, key2):
    # Get mutual items
    mutual_items = {}
    for item in dataset[key1]:
        if item in dataset[key2]:
            mutual_items[item] = 1

    # If there are no ratings in common, return 0
    if len(mutual_items) == 0:
        return 0

    # Sum of the rating squares
    sum1_squares = sum([pow(dataset[key1][item], 2) for item in dataset[key1]])
    sum2_squares = sum([pow(dataset[key2][item], 2) for item in dataset[key2]])

    # Sum of the products
    sum_product = sum([dataset[key1][item] * dataset[key2][item] for item in mutual_items])

    # Calculate score
    numerator = sum_product
    denominator = sqrt(sum1_squares) * sqrt(sum2_squares)
    if denominator == 0:
        return 0

    score = numerator / denominator
    return score


class CollaborativeFiltering:

    # Transform rows into columns and vice versa
    # Transform dataset from user-centric to item-centric and vice-versa
    # Returns the transposed dataset
    @staticmethod
    def __transpose_dataset(dataset):
        transposed_dataset = {}
        for item_i in dataset:
            for item_j in dataset[item_i]:
                transposed_dataset.setdefault(item_j, {})
                transposed_dataset[item_j][item_i] = dataset[item_i][item_j]

        return transposed_dataset

    # Cross validate item-based collaborative filtering
    # "external_similarities" is an optional parameter to pass item-item (user-user) similarities ...
    # These similarities could be computed using some external resources like items' (users') meta-data
    # These similarities should be provided in a dictionary of key01-key02 keys,
    # where key01, key02 are user_ids in UBCF or item_ids in IBCF
    @staticmethod
    def cross_validate(path,
                       item_based=True,
                       k=30,
                       similarity=similarity_cosine,
                       n_folds=10,
                       models_directory='',
                       load_models=False,
                       external_similarities=None,
                       alpha=0.5):
        rmse = 0

        model_name = ''
        if item_based:
            model_name = 'items'
        else:
            model_name = 'users'

        # Read dataset and split it
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (int(row[0]), int(row[1]), float(row[2]))
                dataset.append((user_id, movie_id, rating))

        ''' ***
        # This code generates one similarity model for all folds
        # It is faster but not the best choice,
        # as it uses ratings from the testset in computing similarity
        if not load_models:
            cf = CollaborativeFiltering(k, similarity)
            cf.set_dataset(dataset)
            cf.train(item_based=item_based)
            cf.save_model(models_directory + '/{}_model_f{}_sim_{}.csv'.format(model_name, 'All', similarity.__name__))
        '''

        # Use shuffle=True to shuffle the folds selection
        folds = KFold(n=len(dataset), n_folds=n_folds, shuffle=True)

        fold = 0
        for train_indices, test_indices in folds:
            fold += 1
            training_set = [dataset[i] for i in train_indices]
            test_set = [dataset[i] for i in test_indices]

            print("Fold (%d) started (%s)" % (fold, time.strftime('%y_%m_%d_%H_%M_%S')))
            cf = CollaborativeFiltering(k, similarity)
            cf.set_dataset(training_set)

            # Saving models for later faster use to test the implementation
            # '''***
            if load_models:
                if models_directory:
                    cf.load_model(models_directory + '/{}_model_f{}_sim_{}.csv'.format(model_name, fold, similarity.__name__))
            else:
                cf.train(item_based=item_based)
                if models_directory:
                    cf.save_model(models_directory + '/{}_model_f{}_sim_{}.csv'.format(model_name, fold, similarity.__name__))
            # '''

            '''***'''
            # if models_directory:
                # cf.load_model(models_directory + '/{}_model_f{}_sim_{}.csv'.format(model_name, 'All', similarity.__name__))

            # Inject the external similarities if they were provided
            if external_similarities is not None:
                cf.modify_pairwise_similarity(external_similarities, alpha=alpha)

            cf.predict_missing_ratings(item_based=item_based)
            predict_set = cf.predict_for_set(test_set)

            rmse += mean_squared_error([rec[2] for rec in test_set], [rec[2] for rec in predict_set]) ** 0.5
            print("Fold (%d) finished with accumulated RMSE of (%f) (%s)" % (fold, rmse, time.strftime('%y_%m_%d_%H_%M_%S')))

        return rmse / float(n_folds)

    def cross_validate_item_based(path,
                                  k=30,
                                  similarity=similarity_cosine,
                                  n_folds=10,
                                  models_directory='',
                                  load_models=False,
                                  external_similarities=None,
                                  alpha=0.5):
        CollaborativeFiltering.cross_validate(path=path,
                                              item_based=True,
                                              k=k,
                                              similarity=similarity,
                                              n_folds=n_folds,
                                              models_directory=models_directory,
                                              load_models=load_models,
                                              external_similarities=external_similarities,
                                              alpha=alpha)

    def cross_validate_user_based(path,
                                  k=30,
                                  similarity=similarity_cosine,
                                  n_folds=10,
                                  models_directory='',
                                  load_models=False,
                                  external_similarities=None,
                                  alpha=0.5):
        CollaborativeFiltering.cross_validate(path=path,
                                              item_based=False,
                                              k=k,
                                              similarity=similarity,
                                              n_folds=n_folds,
                                              models_directory=models_directory,
                                              load_models=load_models,
                                              external_similarities=external_similarities,
                                              alpha=alpha)

    # Constructor
    def __init__(self, k=25, similarity=similarity_pearson):
        self.__dataset = {}
        self.__pairwise_similarity = {}
        self.__mean_user_ratings = {}
        self.k = k
        self.similarity = similarity

    # Normalize dataset by subtracting mean user ratings
    def __normalize_dataset(self):
        for user in self.__dataset:
            for item in self.__dataset[user]:
                self.__dataset[user][item] -= self.__mean_user_ratings[user]

    # Denormalize dataset by adding mean user ratings
    def __denormalize_dataset(self):
        for user in self.__dataset:
            for item in self.__dataset[user]:
                self.__dataset[user][item] += self.__mean_user_ratings[user]

    # Set the dataset from a triples list
    # The triples must be in the following format (user, item, rating)
    def set_dataset(self, dataset):
        self.__dataset = {}
        for (user_id, movie_id, rating) in dataset:
            self.__dataset.setdefault(int(user_id), {})
            self.__dataset[int(user_id)][int(movie_id)] = float(rating)

        # Set mean user ratings
        self.__mean_user_ratings = {}
        for user in self.__dataset:
            self.__mean_user_ratings[user] = sum(self.__dataset[user].values()) / len(self.__dataset[user].values())

    # Load dataset from a csv file that is formatted as triples
    # The triples must be in the following format (user, item, rating)
    def load_dataset(self, path):
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id, rating) = (int(row[0]), int(row[1]), float(row[2]))
                dataset.append((user_id, movie_id, rating))

        self.set_dataset(dataset)
        return dataset

    # Calculate pairwise similarity scores
    # user-user similarity for UBCF
    # item-item similarity for IBCF
    def calculate_pairwise_similarity(self, item_based=True):
        self.__pairwise_similarity = {}

        dataset_centered = self.__dataset
        # If the algorithm it item-based collaborative filtering,
        # invert the dataset to be item-centric
        if item_based:
            dataset_centered = CollaborativeFiltering.__transpose_dataset(self.__dataset)

        c = 0
        # key_i, key_j are user_ids in UBCF or item_ids in IBCF
        for key_i in dataset_centered:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Pairwise_Similarity: %d / %d (%s)" % (c, len(dataset_centered), time.strftime('%y_%m_%d_%H_%M_%S')))

            self.__pairwise_similarity.setdefault(key_i, {})
            # Calculate how similar this object to other objects
            for key_j in dataset_centered:
                # If the similarity is calculated before, don't calculate it again
                if key_j in self.__pairwise_similarity:
                    if key_i in self.__pairwise_similarity[key_j]:
                        self.__pairwise_similarity[key_i][key_j] = self.__pairwise_similarity[key_j][key_i]
                        continue

                # If key_i is item_j set the similarity to one
                if key_i == key_j:
                    self.__pairwise_similarity[key_i][key_j] = 1
                    continue

                self.__pairwise_similarity[key_i][key_j] = self.similarity(dataset_centered, key_i, key_j)

    # Train the model
    # This method is simply calling calculate_pairwise_similarity
    def train_item_based(self):
        self.calculate_pairwise_similarity(item_based=True)

    def train_user_based(self):
        self.calculate_pairwise_similarity(item_based=False)

    def train(self, item_based=True):
        self.calculate_pairwise_similarity(item_based=item_based)

    # Save the trained model into a CSV file as triples
    # The triples are in the following format (key01, key02, similarity_score),
    # where key is user_id in UBCF or item_id in IBCF
    # The trained model is the pairwise similarity
    def save_model(self, path):
        with open(path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for key_i in self.__pairwise_similarity:
                for key_j in self.__pairwise_similarity[key_i]:
                    writer.writerow([key_i, key_j, self.__pairwise_similarity[key_i][key_j]])

    # Load a trained model from a CSV file that is formatted as triples
    # The triples must be in the following format (key01, key02, similarity_score),
    # where key is user_id in UBCF or item_id in IBCF
    # The trained model is the pairwise similarity
    def load_model(self, path):
        self.__pairwise_similarity = {}
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                key_i = int(row[0])
                key_j = int(row[1])
                similarity = float(row[2])

                self.__pairwise_similarity.setdefault(key_i, {})
                self.__pairwise_similarity[key_i][key_j] = similarity

    # Predict missing ratings in the dataset
    def predict_missing_ratings_item_based(self):
        # For each item in pairwise_similarity, sort its similar items
        # according to the similarity scores
        pairwise_similarity_sorted = {}
        print("Sorting started (%s)" % (time.strftime('%y_%m_%d_%H_%M_%S')))
        for item in self.__pairwise_similarity:
            pairwise_similarity_sorted[item] = sorted(self.__pairwise_similarity[item].items(),
                                                      key=lambda rec: rec[1],
                                                      reverse=True)
        print("Sorting finished (%s)" % (time.strftime('%y_%m_%d_%H_%M_%S')))

        # Loop over all users
        c = 0
        for user in self.__dataset:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Missing_Ratings: %d / %d (%s)" % (c, len(self.__dataset), time.strftime('%y_%m_%d_%H_%M_%S')))

            # Loop over all items
            for item in pairwise_similarity_sorted:
                # Ignore if this user has already rated this item
                if item in self.__dataset[user]:
                    continue

                neighbours = 0
                weighted_similarity = 0
                similarities_sum = 0
                # Loop over similar items
                for (similar_item, similarity) in pairwise_similarity_sorted[item]:
                    # Check if the similar item is the item itself
                    if similar_item == item:
                        continue

                    # We are only interested in items that have been rated by the user
                    if similar_item not in self.__dataset[user]:
                        continue

                    neighbours += 1
                    # We are only interested in the k nearest neighbours
                    if neighbours > self.k:
                        break

                    weighted_similarity += similarity * self.__dataset[user][similar_item]
                    similarities_sum += similarity

                if similarities_sum > 0:
                    self.__dataset[user][item] = weighted_similarity / similarities_sum

    def predict_missing_ratings_user_based(self):
        # For each user in pairwise_similarity, sort its similar users
        # according to the similarity scores
        pairwise_similarity_sorted = {}
        print("Sorting started (%s)" % (time.strftime('%y_%m_%d_%H_%M_%S')))
        for user in self.__pairwise_similarity:
            pairwise_similarity_sorted[user] = sorted(self.__pairwise_similarity[user].items(),
                                                      key=lambda rec: rec[1],
                                                      reverse=True)
        print("Sorting finished (%s)" % (time.strftime('%y_%m_%d_%H_%M_%S')))

        # Invert the dataset to be item-centric
        dataset_item_centric = CollaborativeFiltering.__transpose_dataset(self.__dataset)

        # Loop over all items
        c = 0
        for item in dataset_item_centric:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0:
                print("Missing_Ratings: %d / %d (%s)" % (c, len(dataset_item_centric), time.strftime('%y_%m_%d_%H_%M_%S')))

            # Loop over all users
            for user in pairwise_similarity_sorted:
                # Ignore if this user has already rated this item
                if user in dataset_item_centric[item]:
                    continue

                neighbours = 0
                weighted_similarity = 0
                similarities_sum = 0
                # Loop over similar users
                for (similar_user, similarity) in pairwise_similarity_sorted[user]:
                    # Check if the similar user is the user itself
                    if similar_user == user:
                        continue

                    # We are only interested in users that have rated this item
                    if similar_user not in dataset_item_centric[item]:
                        continue

                    neighbours += 1
                    # We are only interested in the k nearest neighbours
                    if neighbours > self.k:
                        break

                    weighted_similarity += similarity * dataset_item_centric[item][similar_user]
                    similarities_sum += similarity

                if similarities_sum > 0:
                    self.__dataset[user][item] = weighted_similarity / similarities_sum

    def predict_missing_ratings(self, item_based=True):
        if item_based:
            self.predict_missing_ratings_item_based()
        else:
            self.predict_missing_ratings_user_based()

    # Predict how the user would rate the item in each tuple in the list
    # The tuples must be in one of the following formats (user, item) or (user, item, rating)
    # If the rating is provided it will be overwritten
    def predict_for_set(self, predict_set):
        result = []
        # Remove the rating if it is already provided
        predict_set = [(rec[0], rec[1]) for rec in predict_set]
        for (user, item) in predict_set:
            rating = 0
            if user in self.__dataset:
                if item in self.__dataset[user]:
                    rating = self.__dataset[user][item]
                else:
                    # Set average user ratings in case of any problem
                    rating = self.__mean_user_ratings[user]

            # Post-process rating in case of any problems
            if rating < 1:
                rating = 1

            if rating > 5:
                rating = 5

            result.append((user, item, rating))

        return result

    # Load dataset from a csv file and predicts how the user would rate the item in each tuple in the file
    # The tuples must be in the following format (user, item)
    def predict_for_set_with_path(self, path):
        # Read dataset
        dataset = []
        with open(path, newline='') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='|')
            for row in reader:
                (user_id, movie_id) = (int(row[0]), int(row[1]))
                dataset.append((user_id, movie_id))

        # Predict
        return self.predict_for_set(dataset)

    # Predict how a user would rate an item
    def predict(self, user, item):
        rating = 0
        if user in self.__dataset:
            if item in self.__dataset[user]:
                rating = self.__dataset[user][item]
            else:
                # Set average user ratings in case of any problem
                rating = self.__mean_user_ratings[user]

        # Post-process rating in case of any problems
        if rating < 1:
            rating = 1

        if rating > 5:
            rating = 5

        return rating

    # Modify pairwise similarity by external similarities
    # These similarities could be computed using other resources like the text describing an item (a user)
    # These similarities should be provided in a dictionary of key01-key02 keys,
    # where key01, key02 are user_ids in UBCF or item_ids in IBCF
    # The modification is based on the weighted sum
    # sim = ((1 - alpha) * sim) + (alpha * external_sim)
    # pairwise_similarity should be computed before calling this function
    def modify_pairwise_similarity(self, external_similarities, alpha=0.5):
        for key_i in self.__pairwise_similarity:
            # If key_i doesn't have similarity scores in external_similarities, skip it
            if key_i not in external_similarities:
                continue

            for key_j in self.__pairwise_similarity[key_i]:
                # If key_j doesn't have similarity score with key_i in external_similarities, skip it
                if key_j not in external_similarities[key_i]:
                    continue

                self.__pairwise_similarity[key_i][key_j] = ((1 - alpha) * self.__pairwise_similarity[key_i][key_j]) + \
                                                           (alpha * external_similarities[key_i][key_j])

    # Set the pairwise similarity matrix to some external similarities matrix
    # This function should normally not be used
    # It is just for testing purposes
    def set_pairwise_similarity(self, external_similarities):
        self.__pairwise_similarity = external_similarities

