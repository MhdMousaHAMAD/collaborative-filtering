from recommenders.collaborative_filtering import CollaborativeFiltering
from recommenders.collaborative_filtering import similarity_cosine
from recommenders.collaborative_filtering import similarity_pearson
import csv


# Read dataset
def read_dataset(path):
    dataset = []
    with open(path, newline='') as file:
        reader = csv.reader(file, delimiter='\t', quotechar='|')
        for row in reader:
            (user_id, movie_id, rating) = (row[0], row[1], float(row[2]))
            dataset.append((user_id, movie_id, rating))

    return dataset


# Write dataset
def write_dataset(dataset, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for instance in dataset:
            writer.writerow(instance)

    return dataset


# Cross validation
rmse = CollaborativeFiltering.cross_validate('data/training.dat',
                                             item_based=True,
                                             k=2,
                                             similarity=similarity_cosine,
                                             n_folds=2,
                                             # models_directory='models', # when provided, the models would be saved
                                             models_directory='',
                                             load_models=False)
print("RMSE: %.3f" % rmse)

# Prediction
ibcf = CollaborativeFiltering(k=2, similarity=similarity_pearson)
ibcf.load_dataset('data/training.dat')
ibcf.train(item_based=True)
# Model can be trained and saved and then loaded again without training
# ibcf.save_model('models/model_sim_{}.csv'.format(ibcf.similarity.__name__))
# ibcf.load_model('models/model_sim_{}.csv'.format(ibcf.similarity.__name__))
ibcf.predict_missing_ratings(item_based=True)
predictions = ibcf.predict_for_set_with_path('data/predict.dat')
write_dataset(predictions, 'output/predictions.dat')
print(predictions)