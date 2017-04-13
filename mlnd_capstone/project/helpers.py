import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def display_sample_images(features, labels, n, normalized=False):
    """
    Displays a set of images from the dataset.
    
    Args:
    ====
    features: array of images.
    labels: array of the feature's labels.
    n: int, number of images to display.
    normalized: boolean, if True adjusts the function to work with the transformed data.
    """

    fig = plt.figure(figsize=(16, 16))
    for i in range(n):
        plt.subplot(1,n,i + 1)
        image_num = np.random.randint(0, high=len(labels))
        if normalized:
            image_num = np.random.randint(0, high=10)
            sample_image = np.squeeze(features[image_num], axis=2)
        else:
            sample_image = features[:,:,:,image_num]
        sample_label = labels[image_num]
        
        plt.axis('off')
        plt.title('Label: {}\nFormat: {}\nMin: {:.2f} | Max: {:.2f}'.format(
            sample_label,
            sample_image.shape,
            sample_image.min(),
            sample_image.max()))

        plt.imshow(sample_image)
    plt.show()


def preprocess_images(features, new_size=224):
    """
    Reshapes the dataset and also transform it depending on the selected options.
    
    Args:
    ====
    features: array of RGB images.
    new_size: int, pads the image to the selected size in all sides.
    
    Return:
    =======
    Array of padded RGB images.
    """

    n = features.shape[3]
    result = []
    pad = (new_size - features.shape[0]) / 2
    pads = ((pad, pad), (pad, pad))
    padded_image = np.empty((new_size, new_size, 3))
    for i in range(n):
        image = features[:,:,:,i]
        for j in range(3):
            padded_image[:,:,j] = np.pad(image[:,:,j], pads, 'constant')
        result.append(padded_image)
    return np.array(result)


def one_hot_encode(labels):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    
    Args:
    ====
    labels: List of sample labels
    
    Return:
    =======
    array of one-hot encoded labels
    """
    result = []
    for i in range(len(x)):
        result.append([1 if x[i] == j + 1 else 0 for j in range(10)])
    return np.array(result)


def batch_creator(features, labels, batch_size, val_size=None):
    for start in range(0, features.shape[3], batch_size):
        end = min(start + batch_size, features.shape[3])
        if val_size is not None:
            feat_batch_train, \
            feat_batch_val,\
            lab_batch_train, \
            lab_batch_val = train_test_split(features[:,:,:,start:end],
                labels[start:end],
                test_size=val_size)
            
            yield feat_batch_train, feat_batch_val, lab_batch_train, lab_batch_val
        else:
            yield features[:,:,:,start:end], labels[start:end]