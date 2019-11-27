import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np



from sklearn.cluster import KMeans
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import silhouette_samples, silhouette_score


csv = pd.read_csv('original.csv', header=None)

print('test')
