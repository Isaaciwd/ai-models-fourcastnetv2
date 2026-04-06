import numpy as np
import torch

from ai_models_fourcastnetv2.fourcastnetv2.sfnonet import FourierNeuralOperatorNet
from ai_models_fourcastnetv2.model import FourCastNetv2
from ai_models_fourcastnetv2.model import SensitivityTarget
from ai_models_fourcastnetv2.model import _parse_target_area
from ai_models_fourcastnetv2.model import _channel_sensitivity_scores
from ai_models_fourcastnetv2.model import _total_sensitivity_map


def test_channel_sensitivity_scores_reduces_last_two_axes():
    gradient = np.array(
        [
            [[1.0, -3.0], [2.0, -2.0]],
            [[0.0, 4.0], [0.0, -4.0]],
        ],
        dtype=np.float32,
    )

    scores = _channel_sensitivity_scores(gradient)

    np.testing.assert_allclose(scores, np.array([2.0, 2.0], dtype=np.float32))


def test_channel_sensitivity_scores_accepts_batched_gradients():
    gradient = np.ones((1, 3, 2, 2), dtype=np.float32)

    scores = _channel_sensitivity_scores(gradient)

    np.testing.assert_allclose(scores, np.ones(3, dtype=np.float32))


def test_total_sensitivity_map_reduces_channel_axis():
    gradient = np.ones((1, 3, 2, 2), dtype=np.float32)

    total_map = _total_sensitivity_map(gradient)

    np.testing.assert_allclose(total_map, np.ones((2, 2), dtype=np.float32))


def test_parse_model_args_accepts_sensitivity_options():
    model = FourCastNetv2.__new__(FourCastNetv2)
    args = model.parse_model_args(
        [
            "--sensitivity",
            "--sensitivity-metric",
            "mean",
            "--sensitivity-path",
            "sens.nc",
            "--model-checkpointing",
            "--no-rollout-checkpointing",
        ]
    )

    assert args.sensitivity is True
    assert args.sensitivity_metric == "mean"
    assert args.sensitivity_path == "sens.nc"
    assert args.model_checkpointing is True
    assert args.rollout_checkpointing is False


def test_parse_model_args_accepts_target_options():
    model = FourCastNetv2.__new__(FourCastNetv2)
    args = model.parse_model_args(
        [
            "--target-param",
            "r",
            "--target-level",
            "850",
            "--target-area",
            "50,230,30,245",
        ]
    )

    assert args.target_param == "r"
    assert args.target_level == 850
    assert args.target_area == "50,230,30,245"


def test_parse_model_args_accepts_config_and_summary_options():
    model = FourCastNetv2.__new__(FourCastNetv2)
    args = model.parse_model_args(
        [
            "--sensitivity-config",
            "sensitivity.yaml",
            "--summary-path",
            "summary.json",
        ]
    )

    assert args.sensitivity_config == "sensitivity.yaml"
    assert args.summary_path == "summary.json"


def test_parse_target_area_parses_north_west_south_east():
    assert _parse_target_area("50,230,30,245") == (50.0, 230.0, 30.0, 245.0)


def test_target_name_sanitizes_string():
    from ai_models_fourcastnetv2.model import _target_name

    assert _target_name("R850 West Coast", "fallback") == "r850-west-coast"


def test_default_target_resolves_pressure_level_param():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.target_field = None
    model.target_param = "r"
    model.target_level = 850
    model.target_area = None
    model.sensitivity_metric = "mean-square"

    target = model.default_target()

    assert target.field == "r850"
    assert target.metric == "mean-square"


def test_sensitivity_objective_uses_selected_field_only():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.means = np.zeros((1, len(model.ordering), 1, 1), dtype=np.float32)
    model.stds = np.ones((1, len(model.ordering), 1, 1), dtype=np.float32)
    model.__dict__["device"] = "cpu"
    model.n_lat = 5
    model.n_lon = 8

    output = torch.zeros(1, len(model.ordering), model.n_lat, model.n_lon)
    output[:, model.ordering.index("r850"), :, :] = 3.0
    output[:, model.ordering.index("t850"), :, :] = 20.0

    target = SensitivityTarget(name="r850", field="r850", area=None, metric="mean")
    objective = model.sensitivity_objective(output, target)

    assert torch.isclose(objective, torch.tensor(3.0))


def test_config_targets_parse_multiple_entries():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.sensitivity_metric = "mean-square"

    targets = model.config_targets(
        {
            "targets": [
                {
                    "name": "west-coast-r850",
                    "param": "r",
                    "level": 850,
                    "area": [50, 230, 30, 245],
                    "metric": "mean",
                },
                {
                    "name": "global-t500",
                    "field": "t500",
                },
            ]
        }
    )

    assert len(targets) == 2
    assert targets[0].field == "r850"
    assert targets[0].area == (50.0, 230.0, 30.0, 245.0)
    assert targets[0].metric == "mean"
    assert targets[1].field == "t500"
    assert targets[1].metric == "mean-square"


def test_config_targets_reject_invalid_metric():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.sensitivity_metric = "mean-square"

    try:
        model.config_targets(
            {
                "targets": [
                    {
                        "field": "t500",
                        "metric": "median",
                    }
                ]
            }
        )
    except ValueError as exc:
        assert "Unsupported metric" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid target metric")


def test_parse_target_area_accepts_sequence():
    assert _parse_target_area([50, 230, 30, 245]) == (50.0, 230.0, 30.0, 245.0)


def test_small_fft_model_supports_backward_with_checkpointing():
    model = FourierNeuralOperatorNet(
        spectral_transform="fft",
        img_size=(32, 64),
        scale_factor=2,
        in_chans=4,
        out_chans=4,
        embed_dim=8,
        num_layers=2,
        checkpointing=True,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    state = torch.randn(1, 4, 32, 64, requires_grad=True)
    output = model(state)
    gradient = torch.autograd.grad(output.square().mean(), state)[0]

    assert gradient.shape == state.shape
    assert torch.isfinite(gradient).all()
