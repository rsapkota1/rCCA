
import itertools
from abc import abstractmethod
from typing import Union, Iterable

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import check_random_state, check_is_fitted

from cca_zoo.utils.check_values import _check_views
from scipy.stats import t
#from cca_zoo.probabilisticmodels import ProbabilisticCCA
from scipy.stats import pearsonr
#from _basemodel import _BaseCCA
from cca_zoo.models._base import _BaseCCA

class _ReferenceCCA(_BaseCCA):
    def __init__(self, my_param=1, **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    def pairwise_correlations(self, views: Iterable[np.ndarray],reference=None, **kwargs): #changed to calculate correlation for pairs
        """
        Predicts the correlations between each view for each dimension for the given data using the fit model

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param kwargs: any additional keyword arguments required by the given model
        :return: all_corrs: an array of the pairwise correlations (k,k,self.latent_dims) where k is the number of views
        """
        transformed_views = self.transform(views, **kwargs) #Multiply with weight
        all_corrs = []

        #Normal correlation of first two variables
        for x, y in itertools.product(transformed_views, repeat=2):
            all_corrs.append(
                np.diag(np.corrcoef(x.T, y.T)[: self.latent_dims, self.latent_dims :])
            )

        if reference is not None:
            reference_score = reference
            first_view = transformed_views[0]
            second_view = transformed_views[-1]
           
            combined_tensor = np.hstack((first_view, reference_score))
            corr_coef = np.corrcoef(combined_tensor.T)[-1, :-1]
            #corr_coef, p_value_1 = pearsonr(combined_tensor.T)[-1, :-1]
            #corr_coef, p_value_1 = pearsonr(combined_tensor.T[:-1], combined_tensor.T[-1])
            #corr_coef, p_value_1 = pearsonr(combined_tensor.T[:, :-1], combined_tensor.T[:, -1])


            
            combined_tensor_2 = np.hstack((second_view, reference_score))
            corr_coef_2 = np.corrcoef(combined_tensor_2.T)[-1, :-1]
            #corr_coef_2, p_value_2 = pearsonr(combined_tensor_2.T)[-1, :-1]
            #corr_coef_2, p_value_2= pearsonr(combined_tensor_2.T[:-1], combined_tensor_2.T[-1])

            


        all_corrs = np.array(all_corrs).reshape(
            (len(transformed_views), len(transformed_views), self.latent_dims)
        )


        return [all_corrs ,corr_coef ,corr_coef_2]

    def score(self, views: Iterable[np.ndarray], y=None, reference=None, **kwargs): #4 - Calculate final score
        """
        Returns average correlation in each dimension (averages over all pairs for multiview)

        :param views: list/tuple of numpy arrays or array likes with the same number of rows (samples)
        :param y: unused but needed to integrate with scikit-learn
        """
        if(len(views)==3): #If number of fitted elements is 3, add third variable to reference and remove it
            reference = views[-1]
            views= views[:-1]
        else:
             reference= None
        # by default return the average pairwise correlation in each dimension (for 2 views just the correlation)

        pair_corrs, corrs_1, corrs_2 = self.pairwise_correlations(views, reference=reference, **kwargs) #Compute correlation and final score
        # n views
        n_views = pair_corrs.shape[0]
        # sum all the pairwise correlations for each dimension. Subtract the self correlations. Divide by the number of views. Gives average correlation
        dim_corrs = (
            pair_corrs.sum(axis=tuple(range(pair_corrs.ndim - 1))) - n_views
        ) / (n_views ** 2 - n_views)

        imp_1 = 0.8
        imp_2 = 0.2
        #imp_3 = 0.1

        #all_scores = imp_1* dim_corrs + imp_2* (corrs_1 + imp_3* corrs_2
        all_scores = imp_1* dim_corrs + imp_2* (corrs_1 + corrs_2)/2
  
        

        return all_scores,  dim_corrs, corrs_1,corrs_2

        #return 0.8*(dim_corrs) + 0.2*((corrs_1 + corrs_2)/2)