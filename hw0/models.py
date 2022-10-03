"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(
        self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], sparse=False, embedding_sharing=True
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        self.embedding_sharing = embedding_sharing

        if self.embedding_sharing:
            self.user = ScaledEmbedding(num_users, embedding_dim)
            self.query = ScaledEmbedding(num_items, embedding_dim)
        else:
            # When not sharing embeddings, we learn separate representations for our two tasks.
            self.user_score = ScaledEmbedding(num_users, embedding_dim)
            self.query_score = ScaledEmbedding(num_items, embedding_dim)

            self.user_likelihood = ScaledEmbedding(num_users, embedding_dim)
            self.query_likelihood = ScaledEmbedding(num_items, embedding_dim)

        self.alpha = ZeroEmbedding(num_users, 1)
        self.beta = ZeroEmbedding(num_items, 1)

        self.mlp = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], 1),
        )
        # ********************************************************
        # ********************************************************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        user_bias = self.alpha(user_ids)  # (n x 1)
        item_bias = self.beta(item_ids)  # (n x 1)

        if self.embedding_sharing:
            # Calculate likelihood predictions
            user_embeddings = self.user(user_ids)  # (n x d)
            item_embeddings = self.query(item_ids)  # (n x d)
            predictions = torch.sum(user_embeddings * item_embeddings, dim=1).unsqueeze(1) + user_bias + item_bias

            # Calculate score predictions
            mlp_inputs = torch.cat((user_embeddings, item_embeddings, user_embeddings * item_embeddings), 1)
            score = self.mlp(mlp_inputs)
        else:
            # Calculate likelihood predictions
            user_embeddings = self.user_likelihood(user_ids)  # (n x d)
            item_embeddings = self.query_likelihood(item_ids)  # (n x d)
            predictions = torch.sum(user_embeddings * item_embeddings, dim=1).unsqueeze(1) + user_bias + item_bias

            # Calculate score predictions
            user_embeddings = self.user_score(user_ids)  # (n x d)
            item_embeddings = self.query_score(item_ids)  # (n x d)

            mlp_inputs = torch.cat((user_embeddings, item_embeddings, user_embeddings * item_embeddings), 1)
            score = self.mlp(mlp_inputs)

        predictions = predictions.squeeze()
        score = score.squeeze()
        # ********************************************************
        # ********************************************************
        # ********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")

        return predictions, score
