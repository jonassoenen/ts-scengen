import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import euclidean_distances


def calculate_energy_score_for_daily_matrix_daily_consumption_data(
    probs_per_sample, daily_train_data, daily_test_data
):
    """
    Calculates the energy score for each sample seperately.
    The probs_per_sample is a dataframe, each row is test sample, the columns are a multi-index of (meterID, year, date)
    Train_consumption data each row is a training year
    Test_consumption data each row is the ground truth day of the corresponding test sample

    """
    energy_scores = np.zeros((len(probs_per_sample),))
    for idx, result_series in enumerate(probs_per_sample):
        meterID, date = result_series.name

        # correct test day
        test_day = daily_test_data.loc[(meterID, date), :].to_numpy()

        # training days with non-zero probability
        train_probabilities = result_series.to_numpy()
        train_days = daily_train_data.loc[result_series.index, :]

        # calculate the energy score
        energy_score = calculate_energy_score(train_probabilities, train_days, test_day)

        # return the result
        energy_scores[idx] = energy_score
    return energy_scores


def calculate_energy_score_for_daily_samples(predictions, daily_data_test):
    energy_scores = np.zeros(len(predictions))
    for idx, ((samples, sample_probs), (meterID, truth)) in enumerate(
        zip(predictions, daily_data_test.iterrows())
    ):
        energy_score = calculate_energy_score(sample_probs, samples, truth.to_numpy())
        energy_scores[idx] = energy_score
    return energy_scores


def calculate_energy_score(probs_for_sample, samples, correct_profile):
    """
    samples: 2d numpy array shape (#samples, #dim) , rows are predicted samples
    probs_for_sample: 1d numpy array shape (#samples), for each sample in samples the probability that it will get sampled
    correct_profile: 1d numpy array shape (#dim), the ground truth sample
    """

    def reduce_function(chunk, start):
        sums = (chunk * probs_for_sample).sum(axis=1)
        return sums * probs_for_sample[start : start + sums.shape[0]]

    # because the full distance array can be large, process the chunks seperately into the needed sum
    second_term = 0
    for chunk in pairwise_distances_chunked(samples, reduce_func=reduce_function):
        second_term += chunk.sum(axis=None)
    distances_between_test_and_training_days = euclidean_distances(
        samples, correct_profile.reshape((1, -1))
    ).squeeze()

    first_term = np.sum(probs_for_sample * distances_between_test_and_training_days)

    return first_term - 0.5 * second_term
