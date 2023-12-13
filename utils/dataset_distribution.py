def dataset_distribution(classes, dataset):
    """
    returns a dictionnary distribution for each class
    """
    distrib = {i: 0 for i in range(len(classes))}

    for i in range(dataset.__len__()):
        _, label = dataset.__getitem__(i)
        distrib[label]+=1

    return distrib