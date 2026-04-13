# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import os
from functools import cached_property

import numpy as np
import torch
from ai_models.model import Model
from ai_models.sensitivity import add_sensitivity_parser_arguments
from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import target_slug
from ai_models.sensitivity import SensitivityTarget
from torch.utils.checkpoint import checkpoint

import ai_models_fourcastnetv2.fourcastnetv2 as nvs

LOG = logging.getLogger(__name__)


class FourCastNetv2(Model):
    download_url = "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/{file}"
    download_files = ["weights.tar", "global_means.npy", "global_stds.npy"]

    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    param_sfc = ["10u", "10v", "2t", "sp", "msl", "tcwv", "100u", "100v"]

    param_level_pl = (
        ["t", "u", "v", "z", "r"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    ordering = [
        "10u",
        "10v",
        "100u",
        "100v",
        "2t",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ]

    expver = "sfno"
    supported_attribution_methods = ("gradient", "integrated-gradients")

    def __init__(self, precip_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.n_lat = 721
        self.n_lon = 1440
        self.hour_steps = 6
        self.backbone_channels = len(self.ordering)
        self.checkpoint_path = os.path.join(self.assets, "weights.tar")

        default_sensitivity_path = f"{self.__class__.__name__.lower()}-sensitivity-{self.lead_time:03d}h.nc"
        self.init_sensitivity("fourcastnetv2-small", default_sensitivity_path)

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser(add_help=False)
        add_sensitivity_parser_arguments(parser)
        parser.add_argument(
            "--model-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable block-level checkpointing inside FourCastNet v2. Defaults to on in sensitivity mode.",
        )
        parser.add_argument(
            "--rollout-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable checkpointing across autoregressive forecast steps. Defaults to on in sensitivity mode.",
        )
        return parser.parse_args(args)

    def parse_target_field(self, target_field=None, target_param=None, target_level=None):
        if target_field and target_param:
            raise ValueError("Use either --target-field or --target-param, not both")

        if target_field:
            if target_field not in self.ordering:
                raise ValueError(
                    f"Unknown target field '{target_field}'. Expected one of: {', '.join(self.ordering)}"
                )
            return target_field

        if target_param is None:
            if target_level is not None:
                raise ValueError("--target-level requires --target-param")
            return None

        if target_param in self.param_sfc:
            if target_level is not None:
                raise ValueError(f"Surface target '{target_param}' does not take --target-level")
            return target_param

        pressure_params, pressure_levels = self.param_level_pl
        if target_param in pressure_params:
            if target_level is None:
                raise ValueError(f"Pressure-level target '{target_param}' requires --target-level")
            if target_level not in pressure_levels:
                raise ValueError(
                    f"Unsupported level {target_level} for '{target_param}'. Expected one of: {pressure_levels}"
                )
            target_field = f"{target_param}{target_level}"
            if target_field not in self.ordering:
                raise ValueError(f"Target field '{target_field}' is not available in FourCastNet v2")
            return target_field

        raise ValueError(f"Unsupported target parameter '{target_param}'")

    def default_target(self):
        field = self.parse_target_field(self.target_field, self.target_param, self.target_level)
        area = parse_target_area(self.target_area)
        if area is not None and field is None:
            raise ValueError("--target-area requires --target-field or --target-param")

        name = target_slug(field, "full-state")
        return SensitivityTarget(name=name, field=field, area=area, metric=self.sensitivity_metric)

    def config_targets(self, config):
        targets = config.get("targets")
        if not targets:
            return [self.default_target()]

        parsed_targets = []
        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                raise ValueError("Each target entry must be a mapping")

            field = self.parse_target_field(
                target_field=target.get("field"),
                target_param=target.get("param"),
                target_level=target.get("level"),
            )
            area = parse_target_area(target.get("area"))
            if area is not None and field is None:
                raise ValueError("Target area requires a specific target field")

            metric = target.get("metric", self.sensitivity_metric)
            if metric not in ("mean", "mean-square"):
                raise ValueError(f"Unsupported metric '{metric}' for target {index + 1}")

            name = target_slug(target.get("name"), field or f"target-{index + 1}")
            parsed_targets.append(SensitivityTarget(name=name, field=field, area=area, metric=metric))

        return parsed_targets

    def load_statistics(self):
        path = os.path.join(self.assets, "global_means.npy")
        LOG.info("Loading %s", path)
        self.means = np.load(path)
        self.means = self.means[:, : self.backbone_channels, ...]
        self.means = self.means.astype(np.float32)

        path = os.path.join(self.assets, "global_stds.npy")
        LOG.info("Loading %s", path)
        self.stds = np.load(path)
        self.stds = self.stds[:, : self.backbone_channels, ...]
        self.stds = self.stds.astype(np.float32)

    @cached_property
    def means_torch(self):
        return torch.from_numpy(self.means).to(self.device)

    @cached_property
    def stds_torch(self):
        return torch.from_numpy(self.stds).to(self.device)

    @cached_property
    def ordered_fields(self):
        all_fields = self.all_fields
        all_fields = all_fields.sel(
            param_level=self.ordering, remapping={"param_level": "{param}{levelist}"}
        )
        return all_fields.order_by(
            {"param_level": self.ordering},
            remapping={"param_level": "{param}{levelist}"},
        )

    def load_model(self, checkpoint_file):
        model = nvs.FourierNeuralOperatorNet(checkpointing=self.model_checkpointing)

        checkpoint_data = torch.load(checkpoint_file, map_location=self.device)

        try:
            new_state_dict = {}
            for key, value in checkpoint_data["model_state"].items():
                name = key[7:]
                if name != "ged":
                    new_state_dict[name] = value
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint_data["model_state"])

        model.eval()
        model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def normalise(self, data, reverse=False):
        if torch.is_tensor(data):
            means = self.means_torch.to(dtype=data.dtype)
            stds = self.stds_torch.to(dtype=data.dtype)
        else:
            means = self.means
            stds = self.stds

        if reverse:
            return data * stds + means
        return (data - means) / stds

    @property
    def longitudes(self):
        return np.arange(self.n_lon, dtype=np.float32) * self.grid[1]

    @property
    def latitudes(self):
        return np.linspace(90, -90, self.n_lat, dtype=np.float32)

    def target_weights(self, output, area):
        latitudes = torch.from_numpy(self.latitudes).to(device=output.device, dtype=output.dtype)
        longitudes = torch.from_numpy(self.longitudes).to(device=output.device, dtype=output.dtype)

        lat_weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0)
        lat_mask = torch.ones_like(lat_weights, dtype=torch.bool)
        lon_mask = torch.ones_like(longitudes, dtype=torch.bool)

        if area is not None:
            north, west, south, east = area
            lat_mask = (latitudes <= north) & (latitudes >= south)
            if west <= east:
                lon_mask = (longitudes >= west) & (longitudes <= east)
            else:
                lon_mask = (longitudes >= west) | (longitudes <= east)

        weights = lat_weights[:, None] * lat_mask.to(output.dtype)[:, None] * lon_mask.to(output.dtype)[None, :]

        if not torch.any(weights > 0):
            raise ValueError("Selected target area does not overlap the FourCastNet v2 grid")

        return weights / weights.sum()

    def integrated_gradients_baseline(self, input_state):
        baseline_mode = getattr(self, "ig_baseline", "zero")
        if baseline_mode == "zero":
            return torch.zeros_like(input_state)

        if baseline_mode == "climatology":
            # Means are shape (1, C, 1, 1) and broadcast to input_state shape.
            baseline = torch.as_tensor(self.means, device=input_state.device, dtype=input_state.dtype)
            return baseline.expand_as(input_state)

        raise ValueError(f"Unsupported integrated gradients baseline: {baseline_mode}")

    def integrated_gradients(self, model, input_state, objective_fn):
        steps = int(self.ig_steps)
        baseline = self.integrated_gradients_baseline(input_state).detach()
        delta = input_state - baseline
        accum = torch.zeros_like(input_state)

        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
            objective = objective_fn(interpolated)
            gradient = torch.autograd.grad(objective, interpolated)[0]
            accum = accum + gradient

        average_gradient = accum / float(steps)
        return delta * average_gradient

    def _objective_from_input(self, model, input_state, forecast_steps):
        state = self.normalise(input_state)
        objective_state = None

        for _ in range(forecast_steps):
            output = self.model_step(model, state)
            state = output
            objective_state = output

        if objective_state is None:
            raise ValueError("Sensitivity rollout produced no prediction steps")

        return objective_state

    def sensitivity_objective(self, output, target):
        metric = target.metric

        if target.field is None:
            if metric == "mean":
                return output.mean()
            if metric == "mean-square":
                return output.square().mean()
            raise ValueError(f"Unsupported sensitivity metric: {metric}")

        channel_index = self.ordering.index(target.field)
        target_field = self.normalise(output, reverse=True)[:, channel_index, ...]
        weights = self.target_weights(target_field, target.area)

        if metric == "mean":
            return (target_field * weights).sum()
        if metric == "mean-square":
            return (target_field.square() * weights).sum()
        raise ValueError(f"Unsupported sensitivity metric: {metric}")

    def model_step(self, model, state):
        if self.rollout_checkpointing and torch.is_grad_enabled():
            return checkpoint(model, state, use_reentrant=False)
        return model(state)

    def ordered_input_numpy(self):
        data = self.ordered_fields.to_numpy(dtype=np.float32)
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        return data

    def write_forecast_step(self, output, step):
        denormalised = self.normalise(output.detach().cpu().numpy(), reverse=True)
        for index, field in enumerate(self.ordered_fields):
            self.write(denormalised[0, index, ...], check_nans=True, template=field, step=step)
        return denormalised

    def log_output_statistics(self, output, title):
        LOG.debug("%s: %s", title, output.shape)
        for channel, name in enumerate(self.ordering):
            LOG.debug(
                "    %s mean=%s std=%s min=%s max=%s",
                name,
                np.mean(output[:, channel]),
                np.std(output[:, channel]),
                np.amin(output[:, channel]),
                np.amax(output[:, channel]),
            )

    def forecast_step_count(self):
        if self.lead_time <= 0:
            raise ValueError(f"lead_time must be positive, got {self.lead_time}")
        if self.lead_time % self.hour_steps != 0:
            raise ValueError(
                f"For FourCastNet v2, lead_time must be a multiple of {self.hour_steps} hours; got {self.lead_time}"
            )
        return self.lead_time // self.hour_steps

    def run_forecast(self, model, input_state):
        forecast_steps = self.forecast_step_count()
        state = input_state

        with torch.inference_mode():
            with self.stepper(self.hour_steps) as stepper:
                for forecast_index in range(forecast_steps):
                    output = self.model_step(model, state)
                    state = output

                    if forecast_index == 0 and LOG.isEnabledFor(logging.DEBUG):
                        self.log_output_statistics(
                            output.detach().cpu().numpy(),
                            "Mean/stdev of normalised values",
                        )

                    step = (forecast_index + 1) * self.hour_steps
                    denormalised = self.write_forecast_step(output, step)

                    if forecast_index == 0 and LOG.isEnabledFor(logging.DEBUG):
                        self.log_output_statistics(
                            denormalised,
                            "Mean/stdev of denormalised values",
                        )

                    stepper(forecast_index, step)

    def run_sensitivity(self, model, input_state):
        forecast_steps = self.forecast_step_count()
        input_state = input_state.requires_grad_(True)
        state = self.normalise(input_state)
        objective_state = None

        with self.stepper(self.hour_steps) as stepper:
            for forecast_index in range(forecast_steps):
                output = self.model_step(model, state)
                state = output

                if forecast_index == 0 and LOG.isEnabledFor(logging.DEBUG):
                    self.log_output_statistics(
                        output.detach().cpu().numpy(),
                        "Mean/stdev of normalised values",
                    )

                step = (forecast_index + 1) * self.hour_steps
                denormalised = self.write_forecast_step(output, step)

                if forecast_index == 0 and LOG.isEnabledFor(logging.DEBUG):
                    self.log_output_statistics(
                        denormalised,
                        "Mean/stdev of denormalised values",
                    )

                objective_state = output
                stepper(forecast_index, step)

        objectives = []
        for target in self.sensitivity_manager.targets:
            self.sensitivity_manager.current_target = target
            objectives.append(self.sensitivity_manager.objective(objective_state, target))

        gradients = []
        objective_values = []
        if self.attribution_method == "gradient":
            for index, objective in enumerate(objectives):
                gradient = torch.autograd.grad(
                    objective,
                    input_state,
                    retain_graph=(index < len(objectives) - 1),
                )[0]
                gradients.append(gradient.detach().cpu().numpy())
                objective_values.append(float(objective.detach().cpu()))
        elif self.attribution_method == "integrated-gradients":

            def make_objective_function(target):
                def objective_fn(interpolated_input):
                    output_state = self._objective_from_input(model, interpolated_input, forecast_steps)
                    self.sensitivity_manager.current_target = target
                    return self.sensitivity_manager.objective(output_state, target)

                return objective_fn

            for target in self.sensitivity_manager.targets:
                self.sensitivity_manager.current_target = target
                objective_fn = make_objective_function(target)
                attribution = self.integrated_gradients(model, input_state.detach(), objective_fn)
                objective = objective_fn(input_state)
                gradients.append(attribution.detach().cpu().numpy())
                objective_values.append(float(objective.detach().cpu()))
        else:
            raise ValueError(f"Unsupported attribution method: {self.attribution_method}")

        stacked_gradients = np.stack(gradients, axis=0)
        self.sensitivity_manager.save(stacked_gradients, objective_values)

    def run(self):
        self.load_statistics()

        if self.sensitivity_config and not self.sensitivity:
            self.sensitivity = True

        model = self.load_model(self.checkpoint_path)
        input_state = torch.from_numpy(self.ordered_input_numpy()).to(self.device)

        self.write_input_fields(self.ordered_fields)

        if self.sensitivity:
            self.run_sensitivity(model, input_state)
            return

        input_state = self.normalise(input_state)
        self.run_forecast(model, input_state)


def model(model_version, **kwargs):
    models = {
        "0": FourCastNetv2,
        "small": FourCastNetv2,
        "release": FourCastNetv2,
        "latest": FourCastNetv2,
    }
    return models[model_version](**kwargs)
