"""
Collection of functions
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from typing import Union

class TarNetBase(nn.Module):
    def __init__(self,
            sizes_z: tuple = [2048],
            sizes_y: tuple = [200, 1],
            dropout: float=None,
            bn: bool = False,
            return_prob: bool = False,
        ):
        '''
        Class for TarNet model (PyTorch). First layer is Transformer architecture

        Args:
            sizes_z: tuple, size of hidden layers for shared representation
            sizes_y: tuple, size of hidden layers for outcome prediction
            input_dim: int, input dimension for lstm or transformer
            hidden_dim: int, hidden dimension for lstm or transformer
            dropout: float, dropout rate (default: 0.3)
            bn: bool, whether to use batch normalization (default: False)
                Note that after the first layer everything is the feedforward network.
            return_prob: bool, whether to return the predicted probabilities (default: False)
                If return_prob is True, the model will return the predicted probabilities
                for the outcome prediction. Otherwise, it will return the predicted values.
                When it is true, make sure that the size of the last layer of sizes_y matches 
                the number of classes for classification tasks (e.g., 2 for binary classification).
        '''

        super(TarNetBase, self).__init__()
        self.bn: bool = bn
        self.model_z = self._build_model(sizes_z, dropout) #model for shared representation
        self.model_y1 = self._build_model(sizes_y, dropout) #model for Y(1)
        self.model_y0 = self._build_model(sizes_y, dropout) #model for Y(0)
        self.return_prob = return_prob #whether to return the predicted probabilities

    def _build_model(self, sizes: tuple, dropout: float) -> nn.Sequential:
        # create model by nn.Sequential
        layers = []
        for out_size in sizes:
            layers.append(nn.LazyLinear(out_features=out_size))
            if self.bn:
                layers.append(nn.BatchNorm1d(out_size, track_running_stats=False))
            layers.append(nn.ReLU())
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        if self.bn and dropout is not None:
            layers = layers[:-3]  # remove the last BN, ReLU and Dropout
        elif self.bn == False and dropout is None:
            layers = layers[:-1] # remove the last ReLU
        else:
            layers = layers[:-2] # remove the last ReLU and Dropout
        return nn.Sequential(*layers)

    def forward(
            self,
            inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fr = nn.functional.relu(self.model_z(inputs))
        y0 = self.model_y0(fr)
        y1 = self.model_y1(fr)
        if self.return_prob:
            y0 = torch.softmax(y0, dim = 1)
            y1 = torch.softmax(y1, dim = 1)
        return y0, y1, fr

class SpectralNormClassifier(nn.Module):
    """
    A feed-forward neural network for *multi-class* classification with spectral normalization.
    (Also works for binary classification if num_classes=2).

    This classifier applies spectral normalization to each linear layer, ensuring a
    controlled Lipschitz constant and improving training stability. The network’s
    architecture is a multi-layer perceptron (MLP) that can optionally include
    batch normalization and dropout in each hidden layer.

    Parameters
    ----------
    input_dim : int
        Number of input features in the data (dimension of X).
    hidden_sizes : list of int, optional
        Sizes of the hidden layers. Defaults to [128, 64].
    num_classes : int, optional
        Number of output classes. Defaults to 2 (binary classification).
    n_power_iterations : int, optional
        Number of power iterations for computing the spectral norm in each layer.
        Defaults to 1.
    dropout : float, optional
        Dropout probability for each layer. If 0.0, no dropout is applied. Defaults to 0.0.
    batch_norm : bool, optional
        Whether to add a batch normalization layer after each linear layer. Defaults to False.
    lr : float, optional
        Learning rate for the Adam optimizer. Defaults to 2e-6.
    nepoch : int, optional
        Maximum number of training epochs. Defaults to 20.
    batch_size : int, optional
        Batch size used during training. Defaults to 32.
    patience : int, optional
        Patience (in epochs) for early stopping on the validation set. Defaults to 5.
    min_delta : float, optional
        Minimum improvement in validation loss required to reset patience. Defaults to 1e-4.
    use_scheduler : bool, optional
        Whether to use a learning rate scheduler (e.g., StepLR or ReduceLROnPlateau).
        Defaults to False.
    scheduler_type : str, optional
        Scheduler type: "StepLR" or "ReduceLROnPlateau". Defaults to "ReduceLROnPlateau".
    step_size : int, optional
        Step size for the scheduler. Interpreted differently depending on scheduler_type
        ("StepLR" uses it directly, while "ReduceLROnPlateau" treats it as patience).
        Defaults to 5.
    gamma : float, optional
        Learning rate decay factor used by the scheduler. Defaults to 0.5.
    valid_perc : float, optional
        Proportion of data to use for validation (train/valid split). Defaults to 0.2.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass through the network. Returns logits of shape [batch_size, num_classes].
    fit(X: np.ndarray, y: np.ndarray)
        Train the network on a given dataset, including an internal train/valid split,
        early stopping, and optional learning rate scheduling.
    predict_proba(X: np.ndarray) -> np.ndarray
        Predict class probabilities for each sample. Returns an array
        of shape [n_samples, num_classes].
    predict(X: np.ndarray) -> np.ndarray
        Return the hard predicted class (0..num_classes-1) by argmax over the predicted probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list = [128, 64],
        num_classes: int = 2,
        n_power_iterations: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = False,
        lr: float = 2e-6,
        nepoch: int = 20,
        batch_size: int = 32,
        patience: int = 5,
        min_delta: float = 1e-4,
        use_scheduler: bool = False,
        scheduler_type: str = "ReduceLROnPlateau",
        step_size: int = 5,
        gamma: float = 0.5,
        valid_perc: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.n_power_iterations = n_power_iterations
        self.dropout = dropout
        self.batch_norm = batch_norm
        
        # Training hyperparams
        self.lr = lr
        self.epochs = nepoch
        self.batch_size = batch_size
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # LR scheduling
        self.use_scheduler = use_scheduler
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma
        
        self.valid_perc = valid_perc
        
        # Build the feed-forward MLP with spectral norm
        self.mlp = self._build_mlp()
        
        # Define the optimizer (Adam). Scheduler is set up in .fit()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = None

    def _build_mlp(self) -> nn.Sequential:
        """
        Build the network architecture:
          1) hidden layers (with spectral norm, BN, ReLU, Dropout)
          2) final linear layer with spectral norm => outputs logits for all classes
        """
        layers = []
        in_features = self.input_dim
        
        # Add hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(self._build_layer_block(in_features, hidden_size))
            in_features = hidden_size
        
        # Final output layer with out_dim = num_classes
        final_linear = spectral_norm(
            nn.Linear(in_features, self.num_classes),
            n_power_iterations=self.n_power_iterations
        )
        layers.append(final_linear)
        
        return nn.Sequential(*layers)
    
    def _build_layer_block(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """
        Build a single block = spectral_norm(Linear) -> (optional BN) -> ReLU -> (optional Dropout).
        """
        block = []
        
        # Spectral-normalized Linear
        sn_linear = spectral_norm(
            nn.Linear(in_dim, out_dim),
            n_power_iterations=self.n_power_iterations
        )
        block.append(sn_linear)
        
        # Optional BN
        if self.batch_norm:
            block.append(nn.BatchNorm1d(out_dim))
        
        # Activation
        block.append(nn.ReLU())
        
        # Optional dropout
        if self.dropout > 0:
            block.append(nn.Dropout(self.dropout))
        
        return nn.Sequential(*block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP, returning logits of shape [batch_size, num_classes].
        """
        return self.mlp(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train with automatic train/valid split, early stopping, and optional LR scheduling.
        
        Args:
            X: shape [n_samples, input_dim].
            y: shape [n_samples] containing integer class labels in [0..num_classes-1].
        """
        # Convert inputs to torch Tensors
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).long()  # for CrossEntropyLoss => need LongTensor for classes
        
        dataset = TensorDataset(X_torch, y_torch)
        
        # ----- Create train/valid split -----
        n_total = len(dataset)
        n_valid = int(n_total * self.valid_perc)
        n_train = n_total - n_valid
        
        train_dataset, val_dataset = random_split(dataset, [n_train, n_valid])
        
        # Build DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False)
        
        # Loss function (multi-class)
        criterion = nn.CrossEntropyLoss()
        
        # Set up scheduler if needed
        if self.use_scheduler:
            if self.scheduler_type == "StepLR":
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)
            elif self.scheduler_type == "ReduceLROnPlateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=self.gamma,
                    patience=self.step_size, verbose=True
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {self.scheduler_type}")
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(self.epochs):
            # ----- TRAINING -----
            self.train()
            train_loss_sum = 0.0
            num_examples = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                logits = self.forward(batch_X)        # shape [batch_size, num_classes]
                loss = criterion(logits, batch_y)     # batch_y => shape [batch_size], with class indices
                loss.backward()
                self.optimizer.step()
                
                train_loss_sum += loss.item() * batch_X.size(0)
                num_examples += batch_X.size(0)
            train_loss = train_loss_sum / num_examples
            
            # ----- VALIDATION -----
            self.eval()
            val_loss_sum = 0.0
            val_examples = 0
            with torch.no_grad():
                for val_X, val_y in val_loader:
                    val_logits = self.forward(val_X)
                    vloss = criterion(val_logits, val_y)
                    val_loss_sum += vloss.item() * val_X.size(0)
                    val_examples += val_X.size(0)
            val_loss = val_loss_sum / val_examples
            
            # ----- SCHEDULER STEP -----
            if self.scheduler is not None:
                if self.scheduler_type == "StepLR":
                    self.scheduler.step()
                elif self.scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # ----- EARLY STOPPING -----
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= self.patience:
                print(
                    f"Early stopping on epoch {epoch+1}. "
                    f"Val loss did not improve more than {self.min_delta} "
                    f"for {self.patience} consecutive epochs."
                )
                break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return predicted probabilities for each class, shape [n_samples, num_classes].
        """
        self.eval()
        if isinstance(X, np.ndarray):
            X_torch = torch.from_numpy(X).float()
        else:
            X_torch = X.float()
        
        with torch.no_grad():
            logits = self.forward(X_torch)  # shape [n_samples, num_classes]
            probs = torch.nn.functional.softmax(logits, dim=1)  # shape [n_samples, num_classes]
        return probs.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard predictions = argmax across the classes.
        Shape [n_samples], each entry in [0..num_classes-1].
        """
        proba = self.predict_proba(X)  # shape [n_samples, num_classes]
        return np.argmax(proba, axis=1)


def dml_score(
        t: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        tpred: Union[np.ndarray, torch.Tensor],
        ypred1: Union[np.ndarray, torch.Tensor],
        ypred0: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
    '''
    Calculate influence function for the average treatment effect

    Args:
    - t: np.ndarray or torch.Tensor, treatment
    - y: np.ndarray or torch.Tensor, outcome
    - tpred: np.ndarray or torch.Tensor, predicted treatment
    - ypred1: np.ndarray or torch.Tensor, predicted outcome when treated
    - ypred0: np.ndarray or torch.Tensor, predicted outcome when untreated

    Returns:
    - psi: np.array, influence function for the average treatment effect
    '''

    inputs = [t, y, tpred, ypred1, ypred0]
    for i, input in enumerate(inputs):
        if isinstance(input, torch.Tensor):
            inputs[i] = input.cpu().numpy()
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be a numpy array or a PyTorch tensor")
    t, y, tpred, ypred1, ypred0 = inputs

    t = t.reshape(-1); y = y.reshape(-1); tpred = tpred.reshape(-1); ypred1 = ypred1.reshape(-1); ypred0 = ypred0.reshape(-1)

    psi = ypred1 - ypred0 + t * (y - ypred1) / tpred - (1 - t) * (y - ypred0) / (1 - tpred)

    return psi

def estimate_psi_split(
        fr: Union[np.ndarray, torch.Tensor],
        t: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        y0: Union[np.ndarray, torch.Tensor],
        y1: Union[np.ndarray, torch.Tensor],
        ps_model = SpectralNormClassifier,
        ps_model_params: dict = {},
        trim: list = [0.01, 0.99],
        plot_propensity: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

    '''
    Estimate Propensity Score from latent representation with cross-fitting

    Args:
    - fr: np.ndarray or torch.Tensor, latent representation
    - t: np.ndarray or torch.Tensor, treatment
    - y: np.ndarray or torch.Tensor, outcome
    - y0: np.ndarray or torch.Tensor, outcome when treated
    - y1: np.ndarray or torch.Tensor, outcome when untreated
    - ps_model: propensity score model (default: SpectralNormBinaryClassifier)
    - ps_model_params: dict, hyperparameters for the propensity score model
    - trim: list, trimming quantiles for propensity score (default: [0.01, 0.99])
    - plot_propensity: bool, whether to plot the propensity score (default: False)

    Returns:
    psi: np.array, influence function for the average treatment effect
    '''
    inputs = [fr, t, y, y0, y1]
    for i, input in enumerate(inputs):
        if isinstance(input, torch.Tensor):
            inputs[i] = input.cpu().numpy()
        elif not isinstance(input, np.ndarray):
            raise ValueError("Input must be a numpy array or a PyTorch tensor")
    fr, t, y, y0, y1 = inputs

    model_train = ps_model(**ps_model_params)
    model_test = ps_model(**ps_model_params)

    ind = np.arange(fr.shape[0])
    ind1, ind2 = train_test_split(ind, test_size=0.5, random_state=42)

    #model trained for fold 1 -> predict for fold 2
    model_train.fit(fr[ind1,:], t[ind1])
    tpred2 = model_train.predict_proba(fr[ind2,:])[:,1]
    acc2 = accuracy_score(t[ind2], tpred2.round())
    #model trained for fold 2 -> predict for fold 1
    model_test.fit(fr[ind2,:], t[ind2])
    tpred1 = model_test.predict_proba(fr[ind1,:])[:,1]
    acc1 = accuracy_score(t[ind1], tpred1.round())

    acc = (acc1 + acc2) / 2
    print(f"Accuracy Score of Propensity Score Model: {acc}")

    tpreds = np.zeros(len(t))
    tpreds[ind1] = tpred1; tpreds[ind2] = tpred2

    if plot_propensity:
        ts = np.zeros(len(t))
        ts[ind1] = t[ind1]; ts[ind2] = t[ind2]
        _ = plt.hist(tpreds[ts == 1], alpha = 0.5, label = "Treated", density = True)
        _ = plt.hist(tpreds[ts == 0], alpha = 0.5, label = "Control", density = True)
        plt.xlabel("Estimated Propensity Score")
        plt.ylabel("Density")
        plt.legend(loc = "upper left")
        plt.show()

    if trim is not None:
        tpreds[tpreds < min(trim)] = min(trim)
        tpreds[tpreds > max(trim)] = max(trim)

    psi1 = dml_score(t[ind1], y[ind1], tpreds[ind1], y1[ind1], y0[ind1])
    psi2 = dml_score(t[ind2], y[ind2], tpreds[ind2], y1[ind2], y0[ind2])
    psi = np.append(psi1, psi2)
    return psi, tpreds

def TarNet_loss(
        y_true: torch.Tensor,
        t_true: torch.Tensor,
        y0_pred: torch.Tensor,
        y1_pred: torch.Tensor,
        return_probability: bool = False,
    ) -> torch.Tensor:
    """
    Calculate loss function for TarNet.

    Args:
        y_true: torch.Tensor of shape (N,), containing the true outcome.
                When return_probability=True, y_true should contain class indices.
        t_true: torch.Tensor of shape (N,) or (N, 1) with the treatment indicator (0 or 1).
        y0_pred: torch.Tensor, predicted outcome when untreated.
                 For categorical outcomes, shape (N, num_classes) with probabilities.
        y1_pred: torch.Tensor, predicted outcome when treated.
                 For categorical outcomes, shape (N, num_classes) with probabilities.
        return_probability: bool, if True the predictions are assumed to be probability distributions
                            (from softmax) and a categorical loss is applied;
                            otherwise an MSE loss is used.
    Returns:
        A scalar torch.Tensor representing the loss.
    """
    # Get indices for control (t==0) and treated (t==1) groups.
    T0_indices = (t_true.view(-1) == 0).nonzero(as_tuple=True)[0]
    T1_indices = (t_true.view(-1) == 1).nonzero(as_tuple=True)[0]

    if return_probability:
        # For categorical outcomes, assume y_true contains class indices.
        # Compute the negative log likelihood loss per group.
        if T0_indices.numel() > 0:
            y0_selected = y0_pred[T0_indices]           # (n_control, num_classes)
            y_true_control = y_true[T0_indices].long()    # true class labels for control group
            # Get the probability of the true class for each sample.
            prob_control = y0_selected[torch.arange(y0_selected.size(0)), y_true_control]
            loss0 = -torch.log(prob_control + 1e-8).mean()
        else:
            loss0 = 0.0  # No control group observations; set loss0 to 0.

        if T1_indices.numel() > 0:
            y1_selected = y1_pred[T1_indices]           # (n_treated, num_classes)
            y_true_treated = y_true[T1_indices].long()    # true class labels for treated group
            prob_treated = y1_selected[torch.arange(y1_selected.size(0)), y_true_treated]
            loss1 = -torch.log(prob_treated + 1e-8).mean()
        else:
            loss1 = 0.0  # No treatment group observations; set loss1 to 0.
    else:
        # Use Mean Squared Error loss.
        if T0_indices.numel() > 0:
            loss0 = ((y0_pred.view(-1) - y_true.view(-1))[T0_indices] ** 2).mean()
        else:
            loss0 = 0.0  # No control group observations; set loss0 to 0.
            
        if T1_indices.numel() > 0:
            loss1 = ((y1_pred.view(-1) - y_true.view(-1))[T1_indices] ** 2).mean()
        else:
            loss1 = 0.0  # No treatment group observations; set loss1 to 0.

    return loss0 + loss1