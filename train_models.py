import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Int, PRNGKeyArray
from generate_data import PendulumSimulation
import optax
from models import CNNEmulator


def loss_fn(model, batch):
    inputs, targets = batch
    predictions = model(inputs)
    return jnp.mean((predictions - targets) ** 2)


@eqx.filter_jit
def make_step(
    model: CNNEmulator,
    opt_state: optax.OptState,
    batch: tuple[Float[Array, "batch_size 2 n_res n_res"], Float[Array, "batch_size 1 n_res n_res"]],
    optimizer: optax.GradientTransformation,
) -> tuple:
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    # Update parameters
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def train(
    model: CNNEmulator,
    dataset: tuple[Float[Array, "n_samples 2 n_res n_res"], Float[Array, "n_samples 1 n_res n_res"]],
    batch_size: Int,
    learning_rate: Float,
    num_epochs: Int,
    key: PRNGKeyArray,
) -> CNNEmulator:
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    inputs, targets = dataset
    num_samples = inputs.shape[0]

    print("Training...")
    for epoch in range(num_epochs):
        indices = jax.random.permutation(key, num_samples)
        inputs = inputs[indices]
        targets = targets[indices]

        for i in range(0, num_samples, batch_size):
            batch_inputs = inputs[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]
            batch = (batch_inputs, batch_targets)
            model, opt_state, loss = make_step(model, opt_state, batch, optimizer)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model


IMAGE_SIZE = 64

pendulum = PendulumSimulation(image_size=IMAGE_SIZE)
dataset = pendulum.generate_dataset(5, 9.8, 1.0)

CNNmodel = CNNEmulator(jax.random.PRNGKey(0))
trained_CNNmodel = train(CNNmodel, dataset, 4, 1e-3, 300, jax.random.PRNGKey(1))