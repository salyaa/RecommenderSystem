import numpy as np
from collections import namedtuple
import warnings

class SVD:
    """
    Implementation of matrix factorization using Singular Value Decomposition (SVD) for collaborative filtering.
    This implementation includes support for biased and unbiased models, with configurable learning rates
    and regularization parameters for different components.
    """
    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005, reg_all=.02, lr_bu=None,
                 lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None,
                 reg_pu=None, reg_qi=None, random_state=None, verbose=False):
        """
        Initialize SVD model with the following parameters:
        
        Args:
            n_factors (int): Number of latent factors (default: 100)
            n_epochs (int): Number of iterations for optimization (default: 20)
            biased (bool): Whether to use biased SVD (include user/item biases) (default: True)
            init_mean (float): Mean of initial latent factor matrices (default: 0)
            init_std_dev (float): Standard deviation of initial latent factor matrices (default: 0.1)
            lr_all (float): Global learning rate for all parameters (default: 0.005)
            reg_all (float): Global regularization term for all parameters (default: 0.02)
            lr_bu/lr_bi/lr_pu/lr_qi (float): Specific learning rates for user bias/item bias/user factors/item factors
            reg_bu/reg_bi/reg_pu/reg_qi (float): Specific regularization terms for user bias/item bias/user factors/item factors
            random_state (int): Seed for reproducibility (default: None)
            verbose (bool): Whether to print progress during training (default: False)
        """
        # Initialize model parameters
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        
        # Set learning rates - use specific rates if provided, otherwise use global rate
        self.lr_bu = lr_bu if lr_bu is not None else lr_all  
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        
        # Set regularization terms - use specific terms if provided, otherwise use global term
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        
        self.random_state = random_state
        self.verbose = verbose

        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize model components to None (will be set during fitting)
        self.bu = None  # User biases
        self.bi = None  # Item biases
        self.pu = None  # User factors
        self.qi = None  # Item factors
        self.global_mean = None
        self.trainset = None

    def fit(self, trainset):
        """
        Fit the SVD model to the training data.
        
        Args:
            trainset: Training set containing user-item interactions
            
        Returns:
            self: The fitted model
        """
        self.trainset = trainset
        
        # Calculate global mean rating if using biased model
        if self.biased:
            self.global_mean = np.mean([r for (_, _, r) in trainset.all_ratings()])
        else:
            self.global_mean = 0
            
        # Optimization: Use contiguous memory layout with numpy arrays
        # Initialize biases and latent factors
        self.bu = np.zeros(trainset.n_users, dtype=np.double)  # User biases
        self.bi = np.zeros(trainset.n_items, dtype=np.double)  # Item biases
        self.pu = np.random.normal(self.init_mean, self.init_std_dev,
                                 (trainset.n_users, self.n_factors)).astype(np.double)  # User factors
        self.qi = np.random.normal(self.init_mean, self.init_std_dev,
                                 (trainset.n_items, self.n_factors)).astype(np.double)  # Item factors

        # Optimization: Pre-compute rating counts for users and items
        u_rated_counts = np.zeros(trainset.n_users, dtype=np.int32) 
        i_rated_counts = np.zeros(trainset.n_items, dtype=np.int32)
        for u, i, _ in trainset.all_ratings():
            u_rated_counts[u] += 1
            i_rated_counts[i] += 1

        # Perform SGD for specified number of epochs
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(f"\rProcessing epoch {current_epoch+1}")

            self._run_epoch(u_rated_counts, i_rated_counts)

        return self

    def _run_epoch(self, u_rated_counts, i_rated_counts):
        """
        Run a single epoch of the SGD optimization.
        
        Args:
            u_rated_counts (numpy.array): Array containing number of ratings for each user
            i_rated_counts (numpy.array): Array containing number of ratings for each item
        """
        # Optimization: Vectorized operations for faster computation
        for u, i, r in self.trainset.all_ratings():
            # Calculate prediction error
            dot = np.dot(self.qi[i], self.pu[u])
            err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)

            # Update biases if using biased model
            if self.biased:
                self.bu[u] += self.lr_bu * (err - self.reg_bu * self.bu[u])
                self.bi[i] += self.lr_bi * (err - self.reg_bi * self.bi[i])

            # Update latent factors
            self.pu[u] += self.lr_pu * (err * self.qi[i] - self.reg_pu * self.pu[u])
            self.qi[i] += self.lr_qi * (err * self.pu[u] - self.reg_qi * self.qi[i])

    def predict(self, uid, iid):
        """
        Make rating prediction for a given user-item pair.
        
        Args:
            uid: User ID
            iid: Item ID
            
        Returns:
            Prediction: Named tuple containing prediction details
            
        Raises:
            PredictionImpossible: If user or item is unknown
        """
        # Convert external ids to internal ids
        try:
            u = self.trainset.to_inner_uid(uid)
            i = self.trainset.to_inner_iid(iid)
        except ValueError:
            u = i = None

        if u is None or i is None:
            raise PredictionImpossible('User or item is unknown.')

        # Optimization: Direct prediction calculation
        est = self.global_mean
        if self.biased:
            est += self.bu[u] + self.bi[i]
        est += np.dot(self.qi[i], self.pu[u])

        # Optimization: Clamp prediction to valid rating range [1, 5]
        est = min(5, max(1, est))
        
        details = {}
        return Prediction(uid=uid, iid=iid, r_ui=None, est=est, details=details)

class PredictionImpossible(Exception):
    """Exception raised when a prediction cannot be made"""
    pass


# Named tuple for storing prediction results
Prediction = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])