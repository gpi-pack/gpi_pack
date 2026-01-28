from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import os
from patsy import dmatrix
import pandas as pd


from .TNutil import (
    TarNetBase,
    estimate_psi_split,
    SpectralNormClassifier,
)

from typing import Union, Callable


class TarNet:
    '''
    Wrapper class for TarNet model (Imai and Nakamura, work-in-progress)

    Attributes:
    - self.device: torch.device, device used for training
    - self.epochs: int, number of epochs
    - self.batch_size: int, batch size
    - self.num_workers: int, number of workers for data loader
    - self.train_dataloader: DataLoader, dataloader for training
    - self.valid_dataloader: DataLoader, dataloader for validation
    - self.model: TarNetBase, TarNet model
    - self.optim: torch.optim, optimizer (Adam by default)
    - self.scheduler: torch.optim.lr_scheduler, learning rate scheduler
    - self.loss_f: partial, loss function for TarNet

    Methods:
    - create_dataloaders: create dataloaders for training and validation
    - fit: fit the TarNet model
    - validate_step: validate the model
    - predict: predict the outcome
    '''
    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        architecture_y: list = [1],
        architecture_z: list = [1024],
        conv_layers: list[dict] | None = None,
        conv_activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.3,
        step_size: int = None,
        bn: bool = False,
        patience: int = 5,
        min_delta: float = 0.01,
        model_dir: str = None,
        verbose = True,
        random_state: int = 42,
    ):
        '''
        Initializers of the class

        Args:
        - epochs: int, number of epochs
        - batch_size: int, batch size
        - learning_rate: float, learning rate
        - architecture_y: list, architecture of the outcome model
        - architecture_z: list, architecture of the shared representation model
        - conv_layers: list of convolutional layer specs applied before the shared representation.
                   Example: [{"in_channels":3, "out_channels":32, "kernel_size":3, "padding":1, "pool":{"kernel_size":2}}].
                   Leave as None when inputs are already flattened.
        - conv_activation: callable returning an nn.Module activation for conv blocks (default: nn.ReLU)
        - dropout: float, dropout rate
        - step_size: int, step size for the learning rate scheduler (if None, no scheduler)
        - bn: bool, whether to use batch normalization
        - patience: int, patience for early stopping
        - min_delta: float, minimum delta for early stopping
        - model_dir: str, directory for saving the model
        - verbose: bool, whether to print the device
        - random_state: int, random seed for reproducibility
        '''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: Using {self.device}")
        self.epochs = epochs; self.batch_size = batch_size
        self.train_dataloader = None; self.valid_dataloader = None
        self.model = TarNetBase(
            sizes_z = architecture_z,
            sizes_y = architecture_y,
            dropout=dropout,
            bn=bn,
            conv_layers=conv_layers,
            conv_activation=conv_activation,
        ).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        self.step_size = step_size
        if self.step_size is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.5, patience=5)
        self.loss_f = torch.nn.MSELoss()
        self.valid_loss = 0
        self.patience = patience
        self.min_delta = min_delta
        self.model_dir = model_dir
        if self.model_dir is not None and not os.path.exists(self.model_dir):
            print(f"The directory {self.model_dir} does not exist.")
        self.verbose = verbose
        self.random_state = random_state

    def create_dataloaders(self,
            r_train: Union[np.ndarray, torch.Tensor], 
            r_test: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor], 
            y_test: Union[np.ndarray, torch.Tensor],
            t_train: Union[np.ndarray, torch.Tensor], 
            t_test: Union[np.ndarray, torch.Tensor],
            c_train: Union[np.ndarray, torch.Tensor] = None,
            c_test: Union[np.ndarray, torch.Tensor] = None,
        ):
        '''
        Create dataloader for training and validation

        Args:
        - r_train: np.array or torch.Tensor, training data for internal representation
        - r_test: np.array or torch.Tensor, test data for internal representation
        - y_train: np.array or torch.Tensor, training data for outcome
        - y_test: np.array or torch.Tensor, test data for outcome
        - t_train: np.array or torch.Tensor, training data for treatment
        - t_test: np.array or torch.Tensor, test data for treatment
        - c_train: np.array or torch.Tensor, training data for confounders
        - c_test: np.array or torch.Tensor, test data for confounders
        '''
        if c_train is None and c_test is None:
            inputs = [r_train, r_test, y_train, y_test, t_train, t_test]
            index = 1
        else:
            inputs = [r_train, r_test, c_train, c_test, y_train, y_test, t_train, t_test]
            index = 3
        for i, input in enumerate(inputs):
            if isinstance(input, np.ndarray):
                inputs[i] = torch.Tensor(input)
            elif not isinstance(input, torch.Tensor):
                raise ValueError("Input must be either numpy array or torch.Tensor")

            if i > index: #for y, t
                inputs[i] = inputs[i].reshape(-1, 1)
                
        if c_train is None and c_test is None:
            r_train, r_test, y_train, y_test, t_train, t_test = inputs
            train_dataset = TensorDataset(r_train, t_train, y_train)
            valid_dataset = TensorDataset(r_test, t_test, y_test)
        else:
            r_train, r_test, c_train, c_test, y_train, y_test, t_train, t_test = inputs
            train_dataset = TensorDataset(r_train, c_train, t_train, y_train)
            valid_dataset = TensorDataset(r_test, c_test, t_test, y_test)
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, sampler = RandomSampler(train_dataset))
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, sampler = SequentialSampler(valid_dataset))

    def fit(self,
            R: Union[np.ndarray, torch.Tensor],
            Y: Union[np.ndarray, torch.Tensor],
            T: Union[np.ndarray, torch.Tensor],
            C: Union[np.ndarray, torch.Tensor] = None,
            valid_perc: float = 0.2,
            plot_loss: bool = True,
        ):
        '''
        Fit the TarNet model

        Args:
        - R: np.array or torch.Tensor, internal representation
        - Y: np.array or torch.Tensor, outcome
        - T: np.array or torch.Tensor, treatment
        - C: np.array or torch.Tensor, confounders
        - valid_perc: float, percentage of validation data (from 0 to 1)
        - plot_loss: bool, whether to plot the training and validation loss
        '''
        
        if C is not None:
            R_train, R_test, Y_train, Y_test, T_train, T_test, C_train, C_test = train_test_split(
            R, Y, T, C, test_size=valid_perc, random_state= self.random_state,
            )
            use_confounder = True
            self.create_dataloaders(R_train, R_test, Y_train, Y_test, T_train, T_test, C_train, C_test) #r, y, t, c
        else:
            R_train, R_test, Y_train, Y_test, T_train, T_test = train_test_split(
                R, Y, T, test_size=valid_perc, random_state= self.random_state,
            )
            use_confounder = False
            self.create_dataloaders(R_train, R_test, Y_train, Y_test, T_train, T_test) #r, y, t
        all_training_loss = []; all_valid_loss = []; best_loss = 1e10; epochs_no_improve = 0

        #training loop
        self.model.train()
        for epoch in range(self.epochs):
            loss_list = []
            if self.verbose:
                pbar = tqdm(total=len(self.train_dataloader), desc=f'Training (Epoch {epoch})')

            #Need to double check if the following is correct
            if C is not None:
                for _, (r, c, t, y) in enumerate(self.train_dataloader): #r, c, t, y
                    if self.verbose:
                        pbar.update()
                    self.optim.zero_grad()
                    y_pred, _ = self.model(
                        inputs = r.to(self.device), 
                        treatments = t.to(self.device),
                        confounders = c.to(self.device),
                    ) #y_pred, fr
                    loss = self.loss_f(
                        y_pred,
                        y.to(self.device),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping
                    self.optim.step()
                    loss_list.append(loss.item() / len(y))
            else:
                for _, (r, t, y) in enumerate(self.train_dataloader):
                    if self.verbose:
                        pbar.update()
                    self.optim.zero_grad()
                    y_pred, _ = self.model(
                        inputs = r.to(self.device), 
                        treatments = t.to(self.device),
                    )
                    loss = self.loss_f(
                        y_pred,
                        y.to(self.device),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #gradient clipping
                    self.optim.step()
                    loss_list.append(loss.item() / len(y))
            self.model.eval()
            valid_loss = self.validate_step(use_confounder=use_confounder)
            all_training_loss.append(np.mean(loss_list)); all_valid_loss.append(valid_loss)
            if self.verbose:
                print(f"epoch: {epoch}--------- train_loss: {np.mean(loss_list)} ----- valid_loss: {valid_loss}")
            self.model.train()
            if self.step_size is not None:
                self.scheduler.step(valid_loss) #when using ReduceLROnPlateau

            #save the best model
            if valid_loss + self.min_delta < best_loss:
                if self.model_dir != "" and self.model_dir is not None:
                    torch.save(self.model.state_dict(), f"{self.model_dir}/best_TarNet.pth")
                best_loss = valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            #early stopping options
            if epoch >= 5 and epochs_no_improve >= self.patience:
                print(f"Early stopping! The number of epoch is {epoch}.")
                break

        if self.model_dir != "" and self.model_dir is not None:
            if self.verbose:
                print(f"Loading the model saved at {self.model_dir}...")
            self.model.load_state_dict(torch.load(f"{self.model_dir}/best_TarNet.pth"))

        if plot_loss:
            _ = plt.plot(all_training_loss, label = "Training Loss")
            _ = plt.plot(all_valid_loss, label = "Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def validate_step(self, use_confounder: bool = False) -> torch.Tensor:
        '''
        Method to Validate the model
        '''

        valid_loss = []
        with torch.no_grad():
            if self.verbose:
                pbar = tqdm(total=len(self.valid_dataloader), desc= f'Validating')
            if use_confounder:
                for _, (r, c, t, y) in enumerate(self.valid_dataloader):
                    if self.verbose:
                        pbar.update()
                    y_pred, _ = self.model(
                        inputs = r.to(self.device), 
                        treatments = t.to(self.device),
                        confounders = c.to(self.device),
                    )
                    loss = self.loss_f(
                        y_pred,
                        y.to(self.device),
                    )
                    valid_loss.append(loss / len(y))
            else:
                for _, (r, t, y) in enumerate(self.valid_dataloader): #r t y
                    if self.verbose:
                        pbar.update()
                    y_pred, _ = self.model(inputs = r.to(self.device), treatments = t.to(self.device))
                    loss = self.loss_f(
                        y_pred,
                        y.to(self.device),
                    )
                    valid_loss.append(loss / len(y))
        self.valid_loss = torch.Tensor(valid_loss).mean()
        return self.valid_loss

    def predict(
        self, 
        r: Union[np.ndarray, torch.Tensor],
        t: Union[np.ndarray, torch.Tensor],
        c: Union[np.ndarray, torch.Tensor] = None,
        grad_required = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Predict method for TarNet

        Args:
        r: np.ndarray or torch.Tensor, internal representation
        t: np.ndarray or torch.Tensor, treatment

        Returns:
        y_preds: torch.Tensor, predicted outcome for control under the specified treatment status
        frs: torch.Tensor, deconfounder
        '''
        if c is not None:
            inputs = [r, c, t]
        else:
            inputs = [r, t]
        for i, input in enumerate(inputs):
            if isinstance(input, np.ndarray):
                inputs[i] = torch.Tensor(input)
            elif not isinstance(input, torch.Tensor):
                raise ValueError("Input must be either numpy array or torch.Tensor")
        if c is not None:
            r, c, t = inputs
        else:
            r, t = inputs
        t = t.reshape(-1, 1)  # Ensure treatment is a column vector

        #create dataloader for batching
        if c is not None:
            dataset = TensorDataset(r, c, t)
        else:
            dataset = TensorDataset(r, t)
        dataloader  = DataLoader(dataset, batch_size= self.batch_size)

        y_preds = []; frs = []
        if grad_required:
            if c is not None:
                for r, c, t in tqdm(dataloader, total=len(dataloader), desc = 'Predicting'):
                    y_pred, fr = self.model(
                        inputs = r.to(self.device), 
                        treatments = t.to(self.device),
                        confounders = c.to(self.device),
                    )
                    y_preds.append(y_pred)
                    frs.append(fr)
            else:
                for r, t in tqdm(dataloader, total=len(dataloader), desc = 'Predicting'): #r
                    y_pred, fr = self.model(
                        inputs = r.to(self.device),
                        treatments = t.to(self.device),
                    ) #y0, y1, fr
                    y_preds.append(y_pred)
                    frs.append(fr)
        else:
            with torch.no_grad():
                if c is not None:
                    for r, c, t in tqdm(dataloader, total=len(dataloader), desc = 'Predicting'):
                        y_pred, fr = self.model(
                            inputs = r.to(self.device), 
                            treatments = t.to(self.device),
                            confounders = c.to(self.device),
                        )
                        y_preds.append(y_pred)
                        frs.append(fr)
                else:
                    for r, t in tqdm(dataloader, total=len(dataloader), desc = 'Predicting'): #r
                        y_pred, fr = self.model(
                            inputs = r.to(self.device),
                            treatments = t.to(self.device),
                        ) #y0, y1, fr
                        y_preds.append(y_pred)
                        frs.append(fr)

        # Concatenate all the batched predictions
        y_preds = torch.cat(y_preds)
        frs = torch.cat(frs)

        return y_preds, frs


def estimate_k_ate(
        R: Union[list, np.ndarray],
        Y: Union[list, np.ndarray],
        T: Union[list, np.ndarray],
        C: Union[list, np.ndarray] = None,
        formula_C: str = None,
        data: pd.DataFrame = None,
        K: int = 2,
        valid_perc: float = 0.2,
        plot_propensity: bool = True,
        ps_model = SpectralNormClassifier,
        ps_model_params: dict = {},
        batch_size: int = 32,
        nepoch: int = 200,
        step_size: int = None,
        lr: float = 2e-5,
        cluster: list = None,
        dropout: float = 0.2,
        architecture_y: list = [200, 1],
        architecture_z: list = [2048],
        conv_layers: list[dict] | None = None,
        conv_activation: Callable[[], nn.Module] = nn.ReLU,
        trim: list = [0.01, 0.99],
        bn: bool = False,
        patience: int = 5,
        min_delta: float = 0,
        model_dir: str = None,
        verbose: bool = True,
    ) -> tuple[float, float, float]:
    """
    Estimate the Average Treatment Effect (ATE) using TarNet with k-fold cross-fitting.

    This function trains a TarNet model to estimate potential outcomes under treated (T=1) and untreated (T=0) scenarios, 
    and then computes the ATE. It uses k-fold cross-fitting, which helps reduce overfitting by estimating the
    propensity and outcomes on unseen folds.

    Parameters
    ----------
    R : list or np.ndarray
        A list or NumPy array of hidden states extracted from LLM.
        Shape: (N, d_R) where N is the number of samples and d_R is the dimension of hidden states.
        You can load the stored hidden states using `load_hiddens` function.

    Y : list or np.ndarray
        A list or NumPy array of outcomes, shape: (N,).

    T : list or np.ndarray
        A list or NumPy array of treatments, shape: (N,). Typically binary (0 or 1).

    C : list or np.ndarray, optional
        A matrix of additional confounders, shape: (N, d_C). If provided, these will be
        concatenated to R along axis=1. You can pass either this parameter directly
        or use `formula_c` and `data`.

    formula_c : str, optional
        A Patsy-style formula (e.g., `"conf1 + conf2"`) that specifies how to build
        the confounder matrix from a DataFrame. If this is provided, `data` must also be
        provided, and `C` will be constructed via `dmatrix(formula_c, data)`.
        Intercept is removed from the design matrix.

    data : pandas.DataFrame, optional
        The DataFrame containing the columns used in `formula_c`. If `formula_c` is set,
        this parameter is required. The resulting design matrix is then concatenated
        to R as additional confounders.

    K : int, default=2
        Number of cross-fitting folds (K-fold split).

    valid_perc : float, default=0.2
        Proportion of the training set to use for validation when fitting TarNet in each fold.

    plot_propensity : bool, default=True
        Whether to plot the propensity score distribution in the console or a graphing interface
        (implementation-specific).

    ps_model : object, optional
        A model/classifier used to estimate the propensity score. 
        By default, we use a neural network with Spectral Normalization (to ensure Lipshitz continuity).

    ps_model_params : dict, optional
        Hyperparameters for `ps_model`. For example, `{"input_dim": 2048}` if using a custom model
        requiring an input dimension.

    batch_size : int, default=32
        Batch size for TarNet training.

    nepoch : int, default=200
        Number of epochs to train TarNet.

    step_size : int, optional
        Step size for the learning rate scheduler (if applicable).

    lr : float, default=2e-5
        Learning rate for TarNet.

    cluster : list, optional
        A list indicating cluster membership for each observation. If provided, clustered standard errors will be computed.

    dropout : float, default=0.2
        Dropout rate for TarNet layers.

    architecture_y : list, default=[200, 1]
        List specifying the layer sizes for the outcome heads (treatment-specific networks or final layers).
        For example, [200, 1] means that the outcome model has two hidden layers, the first with 200 units and the second with 1 unit.

    architecture_z : list, default=[2048]
        List specifying the layer sizes for the deconfounder.
        For example, [2048, 2048] means that the deconfounder has two hidden layers, each with 2048 units.

    conv_layers : list of dict, optional
        Specification for convolutional layers applied directly to R when it represents image tensors (N, C, H, W).
        Each dict should at least include `in_channels` (first layer only) and `out_channels`, plus optional Conv2d args
        like `kernel_size`, `stride`, `padding`, and an optional `pool` dict (passed to MaxPool2d/AvgPool2d).

    conv_activation : callable, optional
        Factory returning the activation module inserted after each convolution (default: nn.ReLU).

    trim : list, default=[0.01, 0.99]
        Trimming bounds for the propensity score. 
        Propensity scores outside this range will be replaced with the nearest bound. 

    bn : bool, default=False
        Whether to apply batch normalization in TarNet.

    patience : int, default=5
        Patience for early stopping in TarNet training (number of epochs without improvement).

    min_delta : float, default=0
        Minimum improvement threshold for early stopping.

    model_dir : str, optional
        Directory path where the model checkpoints might be saved. If provided, the best model will be saved here and loaded for predictions.

    verbose : bool, default=True
        Whether to print additional information during training.

    Returns
    -------
    ate_est : float
        The estimated Average Treatment Effect.

    sd_est : float
        An approximate standard error (SE) of the ATE estimate, computed as the standard
        deviation of the cross-fitting estimates divided by the square root of the total
        number of estimates.

    Example Usage
    -----
    >>> import pandas as pd # load pandas module to data frame manipulation
    >>> from TNutil import load_hiddens # load hidden states
    >>> from TarNet import estimate_k # load estimate_k function
    >>> df = pd.DataFrame({
    >>>        'OutcomeVar': [...],   # Y
    >>>        'TreatmentVar': [...], # T
    >>>        'conf1': [...],
    >>>        'conf2': [...],
    >>>    })
    >>> # load hidden states stored as .pt files
    >>> hidden_dir = # directory containing hidden states (e.g., "hidden_last_1.pt" for text indexed 1)
    >>> R = load_hiddens(
    >>>    directory = hidden_dir, 
    >>>    hidden_list= df.index.tolist(), # list of indices for hidden states
    >>>    prefix = "hidden_last_", # prefix of hidden states (e.g., "hidden_last_" for "hidden_last_1.pt")
    >>> )
    >>> # If you want to supply the covariates, you can use either of the following methods:
    >>> # Method 1: supply covariates with a formula and DataFrame
    >>> ate, se = estimate_k_ate(
    >>>    R=R,
    >>>    Y=df['OutcomeVar'].values,
    >>>    T=df['TreatmentVar'].values,
    >>>    formula_c="conf1 + conf2",
    >>>    data=df,
    >>>    K=2,
    >>> )
    >>> print("ATE:", ate, "SE:", se)
    >>> # Method 2: supply covariates using a design matrix
    >>> import numpy as np #load numpy module
    >>> C_mat = np.column_stack([df['conf1'].values, df['conf2'].values])
    >>> ate, se = estimate_k(
    >>>    R=R,
    >>>    Y=df['OutcomeVar'].values,
    >>>    T=df['TreatmentVar'].values,
    >>>    C=C_mat,
    >>>    K=2,
    >>>    ...
    >>> )
    >>> print("ATE:", ate, "SE:", se)
    """

    inputs = [R,Y,T]
    for i, input in enumerate(inputs):
        if isinstance(input, list):
            inputs[i] = np.array(input)
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be either numpy array or list, but not ", type(input))
    R, Y, T = inputs

    if formula_C is not None:
        if data is None:
            raise ValueError("If formula_C is provided, data must be provided as well.")
        formula_C = "-1 + " + formula_C #to remove intercept
        C = dmatrix(formula_C, data, return_type='dataframe').values
    elif C is not None:
        C = np.array(C)
        if C.shape[0] != R.shape[0]:
            raise ValueError("C and R must have the same number of samples")
    else:
        C = None

    psi_list = []
    kf = KFold(n_splits=K, shuffle=True)

    for train_index, test_index in kf.split(Y):
        r_train, r_test = R[train_index], R[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        t_train, t_test = T[train_index], T[test_index]
        if C is not None:
            c_train, c_test = C[train_index], C[test_index]
        model = TarNet(
            epochs= nepoch, learning_rate = lr, batch_size= batch_size,
            architecture_y = architecture_y, architecture_z = architecture_z, dropout=dropout,
            conv_layers=conv_layers, conv_activation=conv_activation,
            step_size= step_size, bn=bn,
            patience= patience, min_delta= min_delta, model_dir=model_dir, verbose=verbose,
        )
        if C is not None:
            model.fit(R = r_train, Y = y_train, T = t_train, C = c_train, valid_perc = valid_perc)
            y0_pred, fr = model.predict(r = r_test, t = np.ones_like(t_test) * 0, c = c_test)
            y1_pred, _ = model.predict(r = r_test, t = np.ones_like(t_test) * 1, c = c_test)
        else:
            model.fit(R = r_train, Y = y_train, T = t_train, valid_perc = valid_perc)
            y0_pred, fr = model.predict(r = r_test, t = np.ones_like(t_test) * 0)
            y1_pred, _ = model.predict(r = r_test, t = np.ones_like(t_test) * 1)


        ps_params = dict(ps_model_params)
        if ps_model == SpectralNormClassifier:
            # Make sure to set the input_dim to the last layer of the shared representation
            # when using the default ps_model (SpectralNormClassifier)
            ps_params.setdefault("input_dim", fr.shape[1])

        #train propensity score purely on test data (use two-fold cross fitting)
        psi, _ = estimate_psi_split(
            fr = fr, t = t_test, y = y_test, y0 = y0_pred, y1 = y1_pred,
            plot_propensity = plot_propensity, trim = trim,
            ps_model= ps_model, ps_model_params= ps_params
        )
        psi_list.extend(psi)

    ate_est = np.mean(psi_list)
    if cluster is None:
        se_est = np.std(psi_list) / np.sqrt(len(psi_list))
    else:
        #clustered standard error
        cluster = np.array(cluster)
        unique_clusters = np.unique(cluster)
        m = len(unique_clusters)
        cluster_sums = []
        for uc in unique_clusters:
            cluster_psi = np.array(psi_list)[cluster == uc]
            cluster_sums.append(np.sum(cluster_psi - ate_est))
        cluster_sums = np.array(cluster_sums)
        se_est = np.sqrt(m / (m - 1) * np.sum(cluster_sums**2)) / len(psi_list)

    print("ATE:", ate_est, " /  SE:", se_est)

    return ate_est, se_est


class TarNetHyperparameterTuner:
    '''
    Class for Hyperparameter tuning for TarNet

    Parameters:
    T: torch.Tensor, treatment variables
    Y: torch.Tensor, outcome variables
    R: torch.Tensor, internal representations
    epoch: int, number of epochs
    batch_size: int, batch size
    valid_perc: float, validation percentage
    learning_rate: list[float, float], range of learning rate
    dropout: list[float, float], range of dropout
    step_size: list[int], list of step sizes
    architecture_y: list[list[str]], list of architectures for Y (e.g., ["[256, 128, 1]", "[256, 128, 64, 1]"])
    architecture_z: list[list[str]], list of architectures for Z (e.g., ["[2048]", "[4096, 2048]"])
    bn: list[bool], list of batch normalization options (e.g., [True, False])
    patience_min: int, minimum value of uniform distribution for the patience for early stopping (default: 5)
    patience_max: int, maximum value of uniform distribution for the patience for early stopping (default: 20)
    conv_layers: list of dict, specification for convolutional layers applied before the shared representation.
                    Example: [{"in_channels":3, "out_channels":32, "kernel_size":3, "padding":1, "pool":{"kernel_size":2}}].
                    Leave as None when inputs are already flattened.
    conv_activation: callable returning an nn.Module activation for conv blocks (default: nn.ReLU

    Example:
    # Load optuna
    import optuna
    # Load data and set hyperparameters
    obj = TarNetHyperparameterTuner(T, Y, R, epoch = 100)
    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(obj.objective, n_trials=100)
    #Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    '''
    def __init__(
        self,
        T, Y, R,
        C: Union[list, np.ndarray] = None,
        formula_C: str = None,
        data: pd.DataFrame = None,
        epoch: list[str] = ["100", "200"],
        batch_size: int = 64,
        valid_perc: float = 0.2,
        learning_rate: list[float, float] = [1e-4, 1e-5],
        dropout: list[float, float] = [0.1, 0.2],
        step_size: list[int] = [5, 10],
        architecture_y: list[list[str]] = ["[1]"],
        architecture_z: list[list[str]] = ["[1024]", "[2048]", "[4096]"],
        conv_layers: list[dict] | None = None,
        conv_activation: Callable[[], nn.Module] = nn.ReLU,
        bn: list[bool] = [True, False],
        patience_min: int = 5,
        patience_max: int = 20,
        model_dir: str = None,
    ):
        #loading data
        self.T = T; self.Y = Y
        if formula_C is not None:
            if data is None:
                raise ValueError("If formula_C is provided, data must be provided as well.")
            formula_C = "-1 + " + formula_C
            C = dmatrix(formula_C, data, return_type='dataframe').values
            R = np.concatenate([R, C], axis=1)
        if C is not None:
            C = np.array(C)
            if C.shape[0] != R.shape[0]:
                raise ValueError("C and R must have the same number of samples")
            R = np.concatenate([R, C], axis=1)
        self.R = R

        #setting hyperparamaters (optional)
        self.epoch = epoch; self.batch_size = batch_size
        self.valid_perc = valid_perc; self.learning_rate = learning_rate
        self.dropout = dropout; self.step_size = step_size
        self.architecture_y = architecture_y; self.architecture_z = architecture_z
        self.bn = bn; self.patience_min = patience_min; self.patience_max = patience_max
        self.model_dir = model_dir
        self.conv_layers = conv_layers
        self.conv_activation = conv_activation

    def get_value_or_suggestion(self, name, values, suggest_method):
        if len(values) == 1:
            return values[0]
        else:
            return suggest_method(name, min(values), max(values))

    def objective(self, trial):
        learning_rate = self.get_value_or_suggestion('learning_rate', self.learning_rate, trial.suggest_loguniform)
        dropout = self.get_value_or_suggestion('dropout', self.dropout, trial.suggest_float)
        epoch = trial.suggest_categorical('epoch', self.epoch)
        step_size = trial.suggest_categorical('step_size', self.step_size)
        architecture_y = trial.suggest_categorical('architecture_y', self.architecture_y)
        architecture_z = trial.suggest_categorical('architecture_z', self.architecture_z)
        bn = trial.suggest_categorical('bn', self.bn)
        patience = trial.suggest_int('patience', self.patience_min, self.patience_max)
        conv_layers = trial.suggest_categorical('conv_layers', self.conv_layers)
        conv_activation = trial.suggest_categorical('conv_activation', self.conv_activation)
        if isinstance(conv_layers, str):
            conv_layers = ast.literal_eval(conv_layers)

        model = TarNet(
            epochs= ast.literal_eval(epoch), #convert string to integer
            batch_size= self.batch_size,
            learning_rate=learning_rate,
            architecture_y=ast.literal_eval(architecture_y), #convert string to list of integers
            architecture_z=ast.literal_eval(architecture_z), #convert string to list of integers
            conv_layers=conv_layers,
            conv_activation=conv_activation,
            dropout=dropout,
            step_size=step_size,
            bn=bn,
            patience = patience,
            min_delta = 0,
            model_dir=self.model_dir,
        )

        model.fit(self.R, self.Y, self.T, valid_perc= self.valid_perc, plot_loss=False)
        validation_loss = model.validate_step().item()
        return validation_loss

def load_hiddens(directory: str, hidden_list: list, prefix: str = None, device: torch.device = "cpu") -> torch.Tensor:
    """
    Function to load hidden representations given a list of file names (without the .pt extension).
    The order of the loaded tensors will follow the order of hidden_list.

    Args:
    - directory: str, directory where hidden representations are stored
    - hidden_list: list of str, list of file names (without .pt) to load
    - prefix: str, prefix of the file names (default: None)
    - device: torch.device, device to load the tensors (default: cpu)

    Returns:
    - tensors: torch.Tensor, 3D tensor of hidden representations (batch_size, seq_len, hidden_dim)
    """
    # Determine map_location
    tensors = []
    for name in tqdm(hidden_list, desc="Loading Tensors"):
        if prefix is not None:
            file_path = os.path.join(directory, f"{prefix}{name}.pt")
        else:
            file_path = os.path.join(directory, f"{name}.pt")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        tensor = torch.load(file_path, map_location=device, weights_only=True).float().cpu()
        tensors.append(tensor)

    tensors = torch.stack(tensors, dim=0).squeeze(1).numpy()
    return tensors