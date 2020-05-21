from sklearn.datasets import load_iris

iris_dataset = load_iris()

############データの確認として###################
print("Keys of iris_dataset: \n{}" .format(iris_dataset.keys()))

print("First five columns of data: {}" .format(iris_dataset['data'][:5]))
print("Shape of data: {}" .format(iris_dataset['data'].shape))

print(iris_dataset['DESCR'][:193] + "\n...")

print("Feature names: {}" .format(iris_dataset['feature_names']))


print("First five columns of target: {}" .format(iris_dataset['target'][:5]))

print("Filename: {}" .format(iris_dataset['filename']))

print("Target names: {}" .format(iris_dataset['target_names']))
