# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Implementing some types of ENN ensembles in JAX."""
from typing import Callable, Optional, Sequence

from enn import base
from enn import utils
from enn.networks import indexers
from enn.networks import priors
import haiku as hk
import jax
import chex
import jax.numpy as jnp


class EnsembleLayer(hk.Module):
    def __init__(
        self,
        ensemble: Sequence[hk.Module],
        priors: Sequence[hk.Module],
        prior_scale: int = 1.0,
        activation=jax.nn.relu,
    ):
        super().__init__(name="layer_ensemble")
        self.ensemble = list(ensemble)
        self.priors = list(priors)
        self.prior_scale = prior_scale
        self.activation = activation

    def __call__(self, inputs: base.Array, index: base.Index) -> base.Array:
        """Index must be a single integer indicating which layer to forward."""
        # during init ensure all module parameters are created.
        _ = [model(inputs) for model in self.ensemble]  # pytype:disable=not-callable
        _ = [model(inputs) for model in self.priors]  # pytype:disable=not-callable
        model_output = self.activation(hk.switch(index, self.ensemble, inputs))

        if len(self.priors) > 0:
            prior_output = self.activation(hk.switch(index, self.priors, inputs))
            output = base.OutputWithPrior(
                model_output, self.prior_scale * prior_output
            ).preds
        else:
            output = model_output

        return output


class LayerEnsembleNetwork(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) and without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        module: hk.Module = hk.Linear,
        correlated=False,
    ):
        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:
            x = hk.Flatten()(inputs)

            layers = [
                EnsembleLayer(
                    [module(output_size) for _ in range(num_ensemble)],
                    [],
                    0.0,
                    jax.nn.relu if i < len(num_ensembles) - 1 else lambda x: x,
                )
                for i, (num_ensemble, output_size) in enumerate(
                    zip(num_ensembles, output_sizes)
                )
            ]

            for layer, index in zip(layers, full_index):
                x = layer(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))
        indexer = indexers.LayerEnsembleIndexer(num_ensembles, correlated)

        def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out

        super().__init__(apply, transformed.init, indexer)


class LayerEnsembleNetworkWithPriors(base.EpistemicNetwork):
    """A layer-ensemble MLP (with flatten) and without any prior."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        prior_scale: float = 1.0,
        module: hk.Module = hk.Linear,
        seed: int = 0,
        correlated=False,
    ):
        def enn_fn(inputs: base.Array, full_index: base.Index) -> base.Output:
            x = hk.Flatten()(inputs)

            layers = [
                EnsembleLayer(
                    [module(output_size) for _ in range(num_ensemble)],
                    [module(output_size) for _ in range(num_ensemble)],
                    prior_scale,
                    jax.nn.relu if i < len(num_ensembles) - 1 else lambda x: x,
                )
                for i, (num_ensemble, output_size) in enumerate(
                    zip(num_ensembles, output_sizes)
                )
            ]

            for layer, index in zip(layers, full_index):
                x = layer(x, index)

            return x

        transformed = hk.without_apply_rng(hk.transform(enn_fn))
        indexer = indexers.LayerEnsembleIndexer(num_ensembles, correlated)

        def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.Output:
            net_out = transformed.apply(params, x, z)
            return net_out

        super().__init__(apply, transformed.init, indexer)


def init_module(
    net_fn, dummy_input: base.Array, rng: int = 0,
) -> Sequence[Callable[[base.Array], base.Array]]:
    transformed = hk.without_apply_rng(hk.transform(net_fn))
    params = transformed.init(next(rng), dummy_input)
    return lambda x, params=params: transformed.apply(params, x)


class LayerEnsembleBranch(hk.Module):
    """Branches a single linear layer to num_ensemble, output_size and indexes it back"""

    def __init__(
        self,
        num_ensemble: int,
        output_size: int,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: str = "layer_ensemble_branch",
    ):
        super().__init__(name=name)
        self.num_ensemble = num_ensemble
        self.output_size = output_size
        if b_init:
            self._b_init = b_init
        else:
            self._b_init = jnp.zeros

    def __call__(self, inputs: jnp.ndarray, index: int = None) -> jnp.ndarray:  # [B, H] -> [B, D, K]
        assert inputs.ndim == 2
        unused_batch, input_size = inputs.shape
        w_init = hk.initializers.TruncatedNormal(stddev=(1.0 / jnp.sqrt(input_size)))
        w = hk.get_parameter(
            "w", [input_size, self.output_size, self.num_ensemble], init=w_init
        )
        b = hk.get_parameter(
            "b", [self.output_size, self.num_ensemble], init=self._b_init
        )
        branched = jnp.einsum("bi,ijk->bjk", inputs, w) + b

        return branched
        # unbranched = 

class LayerEnsembleLinear(hk.Module):
    """Branches a single linear layer to num_ensemble, output_size and indexes it back"""

    def __init__(
        self,
        num_ensemble: int,
        output_size: int,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: str = "layer_ensemble_branch",
    ):
        super().__init__(name=name)
        self.num_ensemble = num_ensemble
        self.output_size = output_size
        if b_init:
            self._b_init = b_init
        else:
            self._b_init = jnp.zeros

    def __call__(self, inputs: jnp.ndarray, index: int = None) -> jnp.ndarray:  # [B, H, K] -> [B, D, K]
        assert inputs.ndim == 3
        unused_batch, input_size, self.num_ensemble = inputs.shape
        w_init = hk.initializers.TruncatedNormal(stddev=(1.0 / jnp.sqrt(input_size)))
        w = hk.get_parameter(
            "w", [input_size, self.output_size, self.num_ensemble], init=w_init
        )
        b = hk.get_parameter(
            "b", [self.output_size, self.num_ensemble], init=self._b_init
        )
        unbranched = jnp.einsum("bik,ijk->bjk", inputs, w) + b

        return unbranched
        # unbranched = 

class LayerEnsembleMLP(hk.Module):
    """Parallel num_ensemble MLPs all with same output_sizes.

  In the first layer, the input is 'branched' to num_ensemble linear layers.
  Then, in subsequent layers it is purely parallel EnsembleLinear.
  """

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        nonzero_bias: bool = True,
        name: str = "layer_ensemble_mlp",
    ):
        super().__init__(name=name)
        self.num_ensembles = num_ensembles
        layers = []
        for index, (num_ensemble, output_size) in enumerate(zip(num_ensembles, output_sizes)):
            if index == 0:
                if nonzero_bias:
                    b_init = hk.initializers.TruncatedNormal(stddev=1)
                else:
                    b_init = jnp.zeros
                layers.append(LayerEnsembleBranch(num_ensemble, output_size, b_init))
            else:
                layers.append(LayerEnsembleLinear(num_ensemble, output_size, jnp.zeros))
        self.layers = tuple(layers)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
        num_layers = len(self.layers)
        out = hk.Flatten()(inputs)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < num_layers - 1:
                out = jax.nn.relu(out)
        return out

def make_einsum_layer_ensemble_mlp_enn(
    output_sizes: Sequence[int], num_ensembles: Sequence[int], nonzero_bias: bool = True,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble ENN.

  This is a specialized implementation for ReLU MLP without a prior network.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    num_ensemble: Integer number of elements in the ensemble.
    nonzero_bias: Whether to make the initial layer bias nonzero.
  Returns:
    EpistemicNetwork as an ensemble of MLP.
  """

    def ensemble_forward(x: base.Array) -> base.OutputWithPrior:
        """Forwards the entire ensemble at given input x."""
        model = LayerEnsembleMLP(output_sizes, num_ensembles, nonzero_bias)
        return model(x)

    transformed = hk.without_apply_rng(hk.transform(ensemble_forward))

    # Apply function selects the appropriate index of the ensemble output.
    def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.OutputWithPrior:
        net_out = transformed.apply(params, x)
        one_hot_index = jax.nn.one_hot(z, num_ensembles[0])
        return jnp.dot(net_out, one_hot_index)

    def init(key: base.RngKey, x: base.Array, z: base.Index) -> hk.Params:
        del z
        return transformed.init(key, x)

    indexer = indexers.EnsembleIndexer(num_ensembles[0])

    return base.EpistemicNetwork(apply, init, indexer)

def make_einsum_layer_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: chex.Array,
    num_ensembles: Sequence[int],
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble with matched prior.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    dummy_input: Example x input for prior initialization.
    num_ensemble: Integer number of elements in the ensemble.
    prior_scale: Float rescaling of the prior MLP.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    seed: integer seed for prior init.

  Returns:
    EpistemicNetwork ENN of the ensemble of MLP with matches prior.
  """

    enn = make_einsum_layer_ensemble_mlp_enn(output_sizes, num_ensembles, nonzero_bias)
    init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    prior_params = enn.init(init_key, dummy_input, jnp.array([]))

    # Apply function selects the appropriate index of the ensemble output.
    def apply_with_prior(
        params: hk.Params, x: base.Array, z: base.Index
    ) -> base.OutputWithPrior:
        ensemble_train = enn.apply(params, x, z)
        ensemble_prior = enn.apply(prior_params, x, z) * prior_scale
        return base.OutputWithPrior(train=ensemble_train, prior=ensemble_prior)

    return base.EpistemicNetwork(apply_with_prior, enn.init, enn.indexer)


class TrueLayerEnsembleBranch(hk.Module):
    """Branches a single linear layer to num_ensemble, output_size and indexes it back"""

    def __init__(
        self,
        num_ensemble: int,
        output_size: int,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: str = "layer_ensemble_branch",
    ):
        super().__init__(name=name)
        self.num_ensemble = num_ensemble
        self.output_size = output_size
        if b_init:
            self._b_init = b_init
        else:
            self._b_init = jnp.zeros

    def __call__(self, inputs: jnp.ndarray, index: int = None) -> jnp.ndarray:  # [B, H] -> [B, D, K] -> [B, D]
        assert inputs.ndim == 2
        unused_batch, input_size = inputs.shape
        w_init = hk.initializers.TruncatedNormal(stddev=(1.0 / jnp.sqrt(input_size)))
        w = hk.get_parameter(
            "w", [input_size, self.output_size, self.num_ensemble], init=w_init
        )
        b = hk.get_parameter(
            "b", [self.output_size, self.num_ensemble], init=self._b_init
        )
        branched = jnp.einsum("bi,ijk->bjk", inputs, w) + b
        unbranched = jnp.dot(branched, one_hot_index)
        one_hot_index = jax.nn.one_hot(index, self.num_ensemble)
        return unbranched

class TrueLayerEnsembleMLP(hk.Module):
    """Parallel num_ensemble MLPs all with same output_sizes.

  In the first layer, the input is 'branched' to num_ensemble linear layers.
  Then, in subsequent layers it is purely parallel EnsembleLinear.
  """

    def __init__(
        self,
        output_sizes: Sequence[int],
        num_ensembles: Sequence[int],
        nonzero_bias: bool = True,
        name: str = "layer_ensemble_mlp",
    ):
        super().__init__(name=name)
        self.num_ensembles = num_ensembles
        layers = []
        for index, (num_ensemble, output_size) in enumerate(zip(num_ensembles, output_sizes)):
            if index == 0:
                if nonzero_bias:
                    b_init = hk.initializers.TruncatedNormal(stddev=1)
                else:
                    b_init = jnp.zeros
                layers.append(LayerEnsembleBranch(num_ensemble, output_size, b_init))
            else:
                layers.append(LayerEnsembleLinear(num_ensemble, output_size, jnp.zeros))
        self.layers = tuple(layers)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
        num_layers = len(self.layers)
        out = hk.Flatten()(inputs)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < num_layers - 1:
                out = jax.nn.relu(out)
        return out

def make_true_einsum_layer_ensemble_mlp_enn(
    output_sizes: Sequence[int], num_ensembles: Sequence[int], nonzero_bias: bool = True,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble ENN.

  This is a specialized implementation for ReLU MLP without a prior network.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    num_ensemble: Integer number of elements in the ensemble.
    nonzero_bias: Whether to make the initial layer bias nonzero.
  Returns:
    EpistemicNetwork as an ensemble of MLP.
  """

    def ensemble_forward(x: base.Array) -> base.OutputWithPrior:
        """Forwards the entire ensemble at given input x."""
        model = TrueLayerEnsembleMLP(output_sizes, num_ensembles, nonzero_bias)
        return model(x)

    transformed = hk.without_apply_rng(hk.transform(ensemble_forward))

    # Apply function selects the appropriate index of the ensemble output.
    def apply(params: hk.Params, x: base.Array, z: base.Index) -> base.OutputWithPrior:
        net_out = transformed.apply(params, x, z)
        return net_out

    def init(key: base.RngKey, x: base.Array, z: base.Index) -> hk.Params:
        del z
        return transformed.init(key, x)

    indexer = indexers.EnsembleIndexer(num_ensembles[0])

    return base.EpistemicNetwork(apply, init, indexer)


def make_true_einsum_layer_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: chex.Array,
    num_ensembles: Sequence[int],
    prior_scale: float = 1.0,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> base.EpistemicNetwork:
    """Factory method to create fast einsum MLP ensemble with matched prior.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    dummy_input: Example x input for prior initialization.
    num_ensemble: Integer number of elements in the ensemble.
    prior_scale: Float rescaling of the prior MLP.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    seed: integer seed for prior init.

  Returns:
    EpistemicNetwork ENN of the ensemble of MLP with matches prior.
  """

    enn = make_true_einsum_layer_ensemble_mlp_enn(output_sizes, num_ensembles, nonzero_bias)
    init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
    prior_params = enn.init(init_key, dummy_input, jnp.array([]))

    # Apply function selects the appropriate index of the ensemble output.
    def apply_with_prior(
        params: hk.Params, x: base.Array, z: base.Index
    ) -> base.OutputWithPrior:
        ensemble_train = enn.apply(params, x, z)
        ensemble_prior = enn.apply(prior_params, x, z) * prior_scale
        return base.OutputWithPrior(train=ensemble_train, prior=ensemble_prior)

    return base.EpistemicNetwork(apply_with_prior, enn.init, enn.indexer)
