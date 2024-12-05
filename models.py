import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray, Int

class CNNEmulator(eqx.Module):
    layer1: eqx.nn.Conv2d
    layer2: eqx.nn.Conv2d

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        # Define convolutional layers
        self.layer1 = eqx.nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1, key=key)
        self.layer2 = eqx.nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, key=key)

    def __call__(self, x: Float[Array, "batch_size in_channels height width"]) -> Float[Array, "batch_size out_channels height width"]:
        batch_size, in_channels, height, width = x.shape

        # Reshape to merge batch and channel dimensions
        x = x.reshape(batch_size * in_channels, height, width)  # Now rank 3

        # Pass through layers
        x = jnp.tanh(self.layer1(x))
        x = jnp.tanh(self.layer2(x))

        # Reshape back to separate batch and channel dimensions
        x = x.reshape(batch_size, -1, height, width)  # Restore batch dimension
        return x