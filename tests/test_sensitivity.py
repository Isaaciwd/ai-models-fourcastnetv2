import numpy as np
import torch

from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import target_slug
from ai_models.sensitivity import SensitivityTarget
from ai_models_fourcastnetv2.fourcastnetv2.sfnonet import FourierNeuralOperatorNet
from ai_models_fourcastnetv2.model import FourCastNetv2


def test_parse_target_area_parses_north_west_south_east():
    assert parse_target_area("50,230,30,245") == (50.0, 230.0, 30.0, 245.0)


def test_parse_target_area_accepts_sequence():
    assert parse_target_area([50, 230, 30, 245]) == (50.0, 230.0, 30.0, 245.0)


def test_target_slug_sanitizes_string():
    assert target_slug("R850 West Coast", "fallback") == "r850-west-coast"


def test_parse_model_args_accepts_sensitivity_options():
    model = FourCastNetv2.__new__(FourCastNetv2)
    args = model.parse_model_args(
        [
            "--sensitivity",
            "--sensitivity-metric",
            "mean",
            "--sensitivity-path",
            "sens.nc",
            "--plot-signed-gradients",
            "--model-checkpointing",
            "--no-rollout-checkpointing",
        ]
    )

    assert args.sensitivity is True
    assert args.sensitivity_metric == "mean"
    assert args.sensitivity_path == "sens.nc"
    assert args.plot_signed_gradients is True
    assert args.model_checkpointing is True
    assert args.rollout_checkpointing is False


def test_parse_model_args_accepts_integrated_gradients_options():
    model = FourCastNetv2.__new__(FourCastNetv2)
    args = model.parse_model_args(
        [
            "--attribution-method",
            "integrated-gradients",
            "--ig-steps",
            "12",
            "--ig-baseline",
            "climatology",
        ]
    )

    assert args.attribution_method == "integrated-gradients"
    assert args.ig_steps == 12
    assert args.ig_baseline == "climatology"


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


def test_integrated_gradients_baseline_zero():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.ig_baseline = "zero"
    x = torch.randn(1, 4, 8, 8)

    baseline = model.integrated_gradients_baseline(x)

    assert torch.allclose(baseline, torch.zeros_like(x))


def test_integrated_gradients_baseline_climatology():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.ig_baseline = "climatology"
    model.means = np.ones((1, 4, 1, 1), dtype=np.float32) * 2.5
    x = torch.randn(1, 4, 8, 8)

    baseline = model.integrated_gradients_baseline(x)

    assert baseline.shape == x.shape
    assert torch.allclose(baseline[:, :, 0, 0], torch.full((1, 4), 2.5))


def test_integrated_gradients_linear_model_matches_expectation():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.ig_steps = 16
    model.ig_baseline = "zero"

    linear = torch.nn.Conv2d(2, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        linear.weight[:] = torch.tensor([[[[3.0]], [[-2.0]]]])

    x = torch.randn(1, 2, 4, 4)

    def objective_fn(inputs):
        return linear(inputs).sum()

    attribution = model.integrated_gradients(linear, x, objective_fn)
    expected = x * torch.tensor([3.0, -2.0]).view(1, 2, 1, 1)

    assert torch.allclose(attribution, expected, atol=1e-4, rtol=1e-4)


def test_forecast_step_count_uses_lead_time_hours():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.hour_steps = 6
    model.lead_time = 48

    assert model.forecast_step_count() == 8


def test_forecast_step_count_rejects_non_multiple():
    model = FourCastNetv2.__new__(FourCastNetv2)
    model.hour_steps = 6
    model.lead_time = 50

    try:
        model.forecast_step_count()
    except ValueError as exc:
        assert "multiple of 6" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-multiple lead_time")
