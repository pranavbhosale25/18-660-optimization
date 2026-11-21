"""Checks for portable actor checkpoints."""

import jax.numpy as jnp
import numpy as np
import pytest

from reinforce_jax import load_mlp_checkpoint, save_mlp_checkpoint


def _actor_params():
    return (
        {
            "w": jnp.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32),
            "b": jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        },
        {
            "w": jnp.asarray([[2.0], [-1.0]], dtype=jnp.float32),
            "b": jnp.asarray([0.125], dtype=jnp.float32),
        },
    )


def test_actor_checkpoint_round_trip_uses_plain_numeric_arrays(tmp_path):
    path = save_mlp_checkpoint(tmp_path / "actor.npz", _actor_params())

    with np.load(path, allow_pickle=False) as archive:
        assert archive["format_version"].dtype.kind in "iu"
        assert archive["layer_0_w"].dtype == np.dtype("float32")

    restored = load_mlp_checkpoint(path, dtype=np.float32)
    for expected_layer, restored_layer in zip(_actor_params(), restored):
        np.testing.assert_array_equal(restored_layer["w"], expected_layer["w"])
        np.testing.assert_array_equal(restored_layer["b"], expected_layer["b"])


def test_actor_checkpoint_rejects_wrong_suffix_and_dtype(tmp_path):
    with pytest.raises(ValueError, match="end in .npz"):
        save_mlp_checkpoint(tmp_path / "actor.bin", _actor_params())

    path = save_mlp_checkpoint(tmp_path / "actor.npz", _actor_params())
    with pytest.raises(ValueError, match="expected float64"):
        load_mlp_checkpoint(path, dtype=np.float64)


def test_actor_checkpoint_rejects_incomplete_archive(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(
        path,
        format_version=np.asarray(1),
        layer_count=np.asarray(1),
        layer_0_w=np.ones((2, 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="fields are incomplete"):
        load_mlp_checkpoint(path)


def test_actor_checkpoint_rejects_numeric_npy_file(tmp_path):
    path = tmp_path / "array-disguised-as-checkpoint.npz"
    with path.open("wb") as stream:
        np.save(stream, np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="is not an NPZ archive"):
        load_mlp_checkpoint(path)
