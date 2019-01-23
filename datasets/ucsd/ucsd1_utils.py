import numpy as np

from datasets import UCSDDatabase


def removed_unlabeled_ucsd1_test_samples(database: UCSDDatabase):
    labeled = [2, 3, 13, 17, 18, 20, 21, 22, 23, 31]
    video_length = 200

    result = np.zeros([len(labeled) * video_length, *database.test_dataset.images.shape[1:]], dtype=np.float32)
    for i, index in enumerate(labeled):
        result[i * 200: (i + 1) * video_length] = \
            database.test_dataset.images[index * video_length: (index + 1) * video_length]
    database.test_dataset.images = result
    database.test_dataset.save_to_npz(force=True)


# db = UCSDDatabase(database_path="/home/zelgunn/Documents/datasets/UCSDped1_updated")
# removed_unlabeled_ucsd1_test_samples(db)
# db.visualize_test_dataset()
