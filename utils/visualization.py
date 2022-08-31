###################################
# TEMP - SHIFTED FROM TSNE TO PCA #
###################################

from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sn
import pandas as pd

# # Say, "the default sans-serif font is COMIC SANS"
# matplotlib.rcParams['font.sans-serif'] = "Consolas"
# # Then, "ALWAYS use sans-serif fonts"
# matplotlib.rcParams['font.family'] = "sans-serif"

# number data to sample for TSNE visualizations
NUM_TSNE = 1000

def cka_plot(save_file, x, y_array, y_label_array):

  cmap = plt.get_cmap('jet')
  marks = ['X', 'o']
  lines = ['solid','dashed']
  colors = cmap(np.linspace(0, 1.0, len(y_label_array)))

  plt.figure(figsize=(8,4))
  for i in range(len(y_array)):
    plt.plot(x, y_array[i], lw=2, color = colors[i], linestyle = lines[i % len(y_array)])
    plt.scatter(x, y_array[i], label = y_label_array[i], color = colors[i], marker = marks[i % len(y_array)],s=50)
  tick_x = x
  tick_x_s = []
  for tick in tick_x:
    tick_x_s.append(str(int(tick)))
  plt.xticks(tick_x, tick_x_s,fontsize=14)
  plt.ylim(0,0.5)
  plt.ylabel('MMD Score', fontweight='bold', fontsize=18)
  plt.xlabel('Training Epochs', fontweight='bold', fontsize=18)
  plt.legend(loc='upper left', prop={'weight': 'bold', 'size': 10})
  plt.grid()
  plt.tight_layout()
  plt.savefig(save_file)  
  plt.close()


def tsne_eval_new(X_in, Y, Ym, Y_pred,  marker_label, save_name, save_name_b, save_name_c, title, num_colors, need_tsne=False):
    from tsne import bh_sne
    
    # tsne embeddings
    if need_tsne:
      X = bh_sne(X_in.astype('float64'))
      X[:,0] = (X[:,0] - min(X[:,0])) / (max(X[:,0]) - min(X[:,0]))
      X[:,1] = (X[:,1] - min(X[:,1])) / (max(X[:,1]) - min(X[:,1]))
    else:
      X = X_in
    first_time = True
    # for color_swap in [True,True,False]:
    for color_swap in [True]:
      # plt.figure(figsize=(6.4, 5.4))

      cmap = plt.get_cmap('jet')
      if color_swap:
        if first_time:
          markers = ["|","_","+",".","x"]
          colors = cmap(np.linspace(0, 1.0, len(markers)))
          colors = ['red','red','red','blue','green']
        else:
          markers = ["_","x"]
      else:
        colors = cmap(np.linspace(0, 1.0, num_colors))
        colors = ['red','red','red','blue','green']
        markers = ["|","_","+",".","x"]

      map_order = [3,1,2,0,4]
      for m in map_order:
        index_m = np.where(Ym == m)[0]
        if len(index_m) > 0:
          index = index_m
          if color_swap and first_time:
            plt.scatter(
                    X[index,0],
                    X[index,1],
                    color=colors[m],
                    marker = markers[m],
                    s=10,
                    label= marker_label[m]
                )
          elif color_swap and not first_time:
            plt.scatter(
                    X[index,0],
                    X[index,1],
                    color=colors[m],
                    marker = markers[0],
                    s=10,
                    label= marker_label[m]
                )
          else:
            plt.scatter(
                    X[index,0],
                    X[index,1],
                    color='k',
                    marker = markers[m],
                    s=10,
                    label= marker_label[m]
            )
      sample_ind = np.random.choice(len(X_in), len(X_in), replace=False)
      for ind in sample_ind:
        if color_swap and first_time:
          plt.scatter(
                      X[ind,0],
                      X[ind,1],
                      color=colors[Ym[ind]],
                      marker=markers[Ym[ind]],
                      s=10,
                  )
        elif color_swap and not first_time:
          if Y[ind] == Y_pred[ind]:
            marker_ind = 0
          else:
            marker_ind = 1
          plt.scatter(
                      X[ind,0],
                      X[ind,1],
                      color=colors[Ym[ind]],
                      marker=markers[marker_ind],
                      s=10,
                  )
        else:
          plt.scatter(
                      X[ind,0],
                      X[ind,1],
                      color=colors[Y[ind]],
                      marker=markers[Ym[ind]],
                      s=10,
                  )
      plt.ylim(-0.35, 1.05)
      plt.xlim(0, 1.05)
      plt.yticks(np.arange(0, 1.1, .1),fontsize=12)
      plt.xticks(np.arange(0, 1.1, .1),fontsize=12)
      
                      
      plt.ylabel("TSNE-1", fontweight="bold", fontsize=12)
      plt.xlabel("TSNE-2", fontweight="bold", fontsize=12)
      # plt.title(
      #     "TSNE - " + str(title),
      #     fontweight="bold",
      #     fontsize=10,
      # )
      plt.grid()
      if len(np.unique(Ym)) > 3:
        plt.legend(markerscale=4, prop={'weight': 'bold', 'size': 8}, loc='lower center')
      else:
        plt.legend(markerscale=5, prop={'weight': 'bold', 'size': 10}, loc='lower center')
      plt.tight_layout()
      if color_swap and first_time:
        plt.savefig(save_name+'-tsne.png')
      elif color_swap and not first_time:
        plt.savefig(save_name_c+'-tsne.png')
      else:
        plt.savefig(save_name_b+'-tsne.png')
      plt.close()
      first_time = False
    pass






def tsne_eval(X_in, Y, save_name, title, num_colors, clusters=None):
  pass

def pca_eval(X_in, Y, save_name, title, num_colors, embedding, clusters=None):
    
    num_colors += 1

    # sample for PCA
    sample_ind = np.random.choice(len(X_in), NUM_TSNE, replace=False)
    X_in = X_in[sample_ind]
    Y = Y[sample_ind]

    # pca embeddings
    if embedding is None:
        embedding = {}
        embedding['scaling'] = StandardScaler().fit(X_in)
        X = embedding['scaling'].transform(X_in)
        embedding['pca'] = PCA(n_components=2).fit(X)
        X = embedding['pca'].transform(X)
        embedding['scaling-vis'] = StandardScaler().fit(X)
        X = embedding['scaling-vis'].transform(X)
    
    # add in clusters
    if clusters is not None:
        X_in = np.concatenate((X_in, clusters), axis=0)
        Y = np.concatenate((Y,-1*np.ones(len(clusters))), axis=0).astype(int)
        
    X = embedding['scaling'].transform(X_in)
    X = embedding['pca'].transform(X)
    X = embedding['scaling-vis'].transform(X)
    
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, num_colors))
    for i in np.unique(Y):
        if i >= 0:
            index = np.where(Y == i)[0]
            plt.scatter(
                    X[index,0],
                    X[index,1],
                    c=[colors[i+1] for j in range(len(index))],
                    s=1.5,
                    label="class " + str(i)
                )
    plt.scatter(
                X[:,0],
                X[:,1],
                c=[colors[Y[j]] for j in range(len(X))],
                s=1.5,
            )
    
    # plot cluster centers
    index = np.where(Y == -1)[0]
    if len(index) > 0:
        plt.scatter(
                X[index,0],
                X[index,1],
                c=[colors[-1] for j in range(len(index))],
                s=50,
                marker='*',
                label="clusters"
            )
    
    plt.ylabel("PCA-1", fontweight="bold", fontsize=12)
    plt.xlabel("PCA-2", fontweight="bold", fontsize=12)
    plt.title(
        "PCA - " + str(title),
        fontweight="bold",
        fontsize=14,
    )
    plt.grid()
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_name+'pca.png')
    plt.close()

    # return pca embedding
    return embedding


def confusion_matrix_vis(y_pred, y_true, save_name, title):

    # confusion matrix
    cm_array = np.asarray(confusion_matrix(y_true, y_pred))

    # csv file
    np.savetxt(save_name+'confusion_matrix.csv', cm_array, delimiter=",", fmt='%.0f')

    # png file
    df_cm = pd.DataFrame(cm_array, index = [str(i) for i in range(len(cm_array))],
                  columns = [str(i) for i in range(len(cm_array))])
    plt.figure(figsize=(7,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}) # font size
    plt.savefig(save_name+'confusion_matrix.png')  
    plt.close()
    pass


# https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
def calculate_cka(X,Y):

    # # linear
    # cka_ = feature_space_linear_cka(X, Y)

    # nonlinear
    cka_ = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))

    return cka_

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)