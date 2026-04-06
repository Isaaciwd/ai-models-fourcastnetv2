# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from functools import cached_property
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from ai_models.model import Model
from ruamel.yaml import YAML
from torch.utils.checkpoint import checkpoint

import ai_models_fourcastnetv2.fourcastnetv2 as nvs

LOG = logging.getLogger(__name__)


def _parse_target_area(value):
    if value is None:
        return None

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        raise ValueError("Target area must be a comma-delimited string or a 4-item list")

    if len(parts) != 4:
        raise ValueError("Target area must be 'north,west,south,east'")

    north, west, south, east = (float(part) for part in parts)
    if north < south:
        raise ValueError("Target area must satisfy north >= south")

    return (north, west % 360.0, south, east % 360.0)


def _target_name(value, fallback):
    if value is None:
        value = fallback
    value = str(value).strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or fallback


def _channel_sensitivity_scores(gradient):
    if gradient.ndim == 4:
        gradient = gradient[0]
    return np.abs(gradient).mean(axis=(-2, -1))


def _total_sensitivity_map(gradient):
    if gradient.ndim == 4:
        gradient = gradient[0]
    return np.abs(gradient).mean(axis=0)


def _channel_maps(gradient):
    if gradient.ndim == 4:
        return gradient[0]
    return gradient


@dataclass
class SensitivityTarget:
    name: str
    field: Optional[str]
    area: Optional[Tuple[float, float, float, float]]
    metric: str


class FourCastNetv2(Model):
    # Download
    download_url = "https://get.ecmwf.int/repository/test-data/ai-models/fourcastnetv2/small/{file}"
    download_files = ["weights.tar", "global_means.npy", "global_stds.npy"]

    # Input
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

    # Output
    expver = "sfno"

    yaml_loader = YAML(typ="safe")

    def __init__(self, precip_flag=False, **kwargs):
        super().__init__(**kwargs)

        self.n_lat = 721
        self.n_lon = 1440
        self.hour_steps = 6

        self.backbone_channels = len(self.ordering)

        self.checkpoint_path = os.path.join(self.assets, "weights.tar")

        if self.model_checkpointing is None:
            self.model_checkpointing = self.sensitivity

        if self.rollout_checkpointing is None:
            self.rollout_checkpointing = self.sensitivity

        if self.plot_sensitivity is None:
            self.plot_sensitivity = self.sensitivity

        if self.plot_top_k < 0:
            raise ValueError("--plot-top-k must be non-negative")

        if self.sensitivity_path is None:
            self.sensitivity_path = (
                f"{self.__class__.__name__.lower()}-sensitivity-{self.lead_time:03d}h.nc"
            )

        if self.summary_path is None:
            self.summary_path = os.path.splitext(self.sensitivity_path)[0] + ".json"

        if self.plot_prefix is None and self.sensitivity_path is not None:
            self.plot_prefix = os.path.splitext(self.sensitivity_path)[0]

        self.plot_area_bounds = None
        self.targets = [self.default_target()]
        self.config = None
        self.config_path = None
        self.current_target = self.targets[0]
        self._warned_missing_cartopy = False

        if self.sensitivity_config:
            self.load_sensitivity_config(self.sensitivity_config)

        if self.plot_area is not None:
            self.plot_area_bounds = _parse_target_area(self.plot_area)

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--sensitivity-config",
            help="YAML file defining sensitivity targets, output, plotting, and optional run settings.",
        )
        parser.add_argument(
            "--sensitivity",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Backpropagate a scalar objective from the final forecast state to the input state.",
        )
        parser.add_argument(
            "--sensitivity-metric",
            choices=("mean", "mean-square"),
            default="mean-square",
            help="Default scalar objective for configured targets.",
        )
        parser.add_argument(
            "--sensitivity-path",
            help="Path to the NetCDF file written for input sensitivities.",
        )
        parser.add_argument(
            "--summary-path",
            help="Optional JSON summary path. Defaults to sensitivity path with .json suffix.",
        )
        parser.add_argument(
            "--target-field",
            help="Final forecast field to target, using native names like r850 or 2t.",
        )
        parser.add_argument(
            "--target-param",
            help="Forecast parameter to target, such as r, t, u, v, z, 2t, or tcwv.",
        )
        parser.add_argument(
            "--target-level",
            type=int,
            help="Pressure level for --target-param when targeting pressure-level fields.",
        )
        parser.add_argument(
            "--target-area",
            help="Lat/lon selection as north,west,south,east in degrees on the model grid.",
        )
        parser.add_argument(
            "--plot-sensitivity",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Write PNG plots for the saved sensitivity outputs. Defaults to on in sensitivity mode.",
        )
        parser.add_argument(
            "--plot-top-k",
            type=int,
            default=6,
            help="Number of top input channels to include in the summary plot.",
        )
        parser.add_argument(
            "--plot-area",
            help="Lat/lon plotting region as north,west,south,east in degrees.",
        )
        parser.add_argument(
            "--plot-prefix",
            help="Prefix for generated sensitivity plot filenames.",
        )
        parser.add_argument(
            "--plot-coastlines",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Draw continent coastlines in sensitivity plots when cartopy is available.",
        )
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
        area = _parse_target_area(self.target_area)
        if area is not None and field is None:
            raise ValueError("--target-area requires --target-field or --target-param")

        name = _target_name(field, "full-state")
        return SensitivityTarget(name=name, field=field, area=area, metric=self.sensitivity_metric)

    def config_targets(self, config):
        targets = config.get("targets")
        if not targets:
            return [self.default_target()]

        parsed_targets: List[SensitivityTarget] = []
        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                raise ValueError("Each target entry must be a mapping")

            field = self.parse_target_field(
                target_field=target.get("field"),
                target_param=target.get("param"),
                target_level=target.get("level"),
            )
            area = _parse_target_area(target.get("area"))
            if area is not None and field is None:
                raise ValueError("Target area requires a specific target field")

            metric = target.get("metric", self.sensitivity_metric)
            if metric not in ("mean", "mean-square"):
                raise ValueError(f"Unsupported metric '{metric}' for target {index + 1}")

            name = _target_name(target.get("name"), field or f"target-{index + 1}")
            parsed_targets.append(SensitivityTarget(name=name, field=field, area=area, metric=metric))

        return parsed_targets

    def load_sensitivity_config(self, path):
        with open(path) as file_handle:
            config = self.yaml_loader.load(file_handle) or {}

        if not isinstance(config, dict):
            raise ValueError("Sensitivity config root must be a mapping")

        run_cfg = config.get("run", {})
        if run_cfg:
            lead_time = run_cfg.get("lead_time")
            if lead_time is not None:
                self.lead_time = int(lead_time)

        output_cfg = config.get("output", {})
        if output_cfg:
            self.sensitivity_path = output_cfg.get("path", self.sensitivity_path)
            self.summary_path = output_cfg.get(
                "summary_path", os.path.splitext(self.sensitivity_path)[0] + ".json"
            )

        plotting_cfg = config.get("plotting", {})
        if plotting_cfg:
            if "enabled" in plotting_cfg:
                self.plot_sensitivity = bool(plotting_cfg["enabled"])
            if "top_k" in plotting_cfg:
                self.plot_top_k = int(plotting_cfg["top_k"])
            if "prefix" in plotting_cfg:
                self.plot_prefix = plotting_cfg["prefix"]
            if "area" in plotting_cfg:
                self.plot_area_bounds = _parse_target_area(plotting_cfg["area"])
            if "coastlines" in plotting_cfg:
                self.plot_coastlines = bool(plotting_cfg["coastlines"])

        if self.plot_top_k < 0:
            raise ValueError("plotting.top_k must be non-negative")

        if self.plot_prefix is None:
            self.plot_prefix = os.path.splitext(self.sensitivity_path)[0]

        self.targets = self.config_targets(config)
        if self.plot_area_bounds is None and self.targets:
            self.plot_area_bounds = self.targets[0].area

        self.current_target = self.targets[0]
        self.config = config
        self.config_path = path
        self.sensitivity = True

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

        model.zero_grad()
        # Load weights

        checkpoint = torch.load(checkpoint_file, map_location=self.device)

        weights = checkpoint["model_state"]
        drop_vars = ["module.norm.weight", "module.norm.bias"]
        weights = {k: v for k, v in weights.items() if k not in drop_vars}

        # Make sure the parameter names are the same as the checkpoint
        # need to use strict = False to avoid this error message when
        # using sfno_76ch::
        # RuntimeError: Error(s) in loading state_dict for Wrapper:
        # Missing key(s) in state_dict: "module.trans_down.weights",
        # "module.itrans_up.pct",
        try:
            # Try adding model weights as dictionary
            new_state_dict = dict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]
                if name != "ged":
                    new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        except Exception:
            model.load_state_dict(checkpoint["model_state"])

        # Set model to eval mode and return
        model.eval()
        model.to(self.device)

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        return model

    def normalise(self, data, reverse=False):
        """Normalise data using pre-saved global statistics"""
        if torch.is_tensor(data):
            means = self.means_torch.to(dtype=data.dtype)
            stds = self.stds_torch.to(dtype=data.dtype)
        else:
            means = self.means
            stds = self.stds

        if reverse:
            new_data = data * stds + means
        else:
            new_data = (data - means) / stds
        return new_data

    @property
    def longitudes(self):
        return np.arange(self.n_lon, dtype=np.float32) * self.grid[1]

    @property
    def latitudes(self):
        return np.linspace(90, -90, self.n_lat, dtype=np.float32)

    @cached_property
    def target_area_bounds(self):
        return self.current_target.area

    @cached_property
    def target_channel_name(self):
        return self.current_target.field

    @cached_property
    def target_channel_index(self):
        if self.target_channel_name is None:
            return None
        return self.ordering.index(self.target_channel_name)

    def clear_target_caches(self):
        for key in ("target_area_bounds", "target_channel_name", "target_channel_index"):
            self.__dict__.pop(key, None)

    @cached_property
    def target_summary(self):
        if not self.targets:
            return None

        entries = []
        for target in self.targets:
            entry = {
                "name": target.name,
                "field": target.field,
                "metric": target.metric,
            }
            if target.area is not None:
                north, west, south, east = target.area
                entry["area"] = {
                    "north": north,
                    "west": west,
                    "south": south,
                    "east": east,
                }
            entries.append(entry)

        if len(entries) == 1:
            return entries[0]
        return entries

    def scalar_objective(self, output):
        if self.sensitivity_metric == "mean":
            return output.mean()
        if self.sensitivity_metric == "mean-square":
            return output.square().mean()
        raise ValueError(f"Unsupported sensitivity metric: {self.sensitivity_metric}")

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

    def total_sensitivity_peak(self, total_map):
        peak_index = np.unravel_index(np.argmax(total_map), total_map.shape)
        lat_index, lon_index = peak_index
        return {
            "latitude": float(self.latitudes[lat_index]),
            "longitude": float(self.longitudes[lon_index]),
            "value": float(total_map[lat_index, lon_index]),
        }

    def add_target_area_patch(self, axes, data_crs=None):
        area = self.plot_area_bounds if self.plot_area_bounds is not None else self.target_area_bounds
        if area is None:
            return

        import matplotlib.patches as patches

        north, west, south, east = area
        spans = [(west, east)] if west <= east else [(west, 360.0), (0.0, east)]
        for span_west, span_east in spans:
            patch_kwargs = {}
            if data_crs is not None:
                patch_kwargs["transform"] = data_crs

            axes.add_patch(
                patches.Rectangle(
                    (span_west, south),
                    span_east - span_west,
                    north - south,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.5,
                    linestyle="--",
                    **patch_kwargs,
                )
            )

    def maybe_add_coastlines(self, axes):
        if not self.plot_coastlines:
            return

        try:
            import cartopy.feature as cfeature

            axes.coastlines(color="black", linewidth=0.6)
            axes.add_feature(cfeature.BORDERS, linewidth=0.3)
        except Exception:
            if not self._warned_missing_cartopy:
                LOG.warning("Cartopy is not available; plotting without coastline overlays")
                self._warned_missing_cartopy = True

    def apply_plot_limits(self, axes, data_crs=None):
        area = self.plot_area_bounds
        if area is None:
            return

        north, west, south, east = area
        if data_crs is not None and hasattr(axes, "set_extent"):
            if west <= east:
                axes.set_extent([west, east, south, north], crs=data_crs)
            else:
                axes.set_extent([0, 360, south, north], crs=data_crs)
            return

        if west <= east:
            axes.set_xlim(west, east)
        else:
            axes.set_xlim(0, 360)
        axes.set_ylim(south, north)

    def write_sensitivity_plots(self, gradient, top_channels):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        projection = None
        data_crs = None
        if self.plot_coastlines:
            try:
                import cartopy.crs as ccrs

                projection = ccrs.PlateCarree(central_longitude=180)
                data_crs = ccrs.PlateCarree()
            except Exception:
                if not self._warned_missing_cartopy:
                    LOG.warning("Cartopy is not available; plotting without coastline overlays")
                    self._warned_missing_cartopy = True

        total_map = _total_sensitivity_map(gradient)
        total_path = f"{self.plot_prefix}-total.png"
        extent = (0.0, 360.0, float(self.latitudes[-1]), float(self.latitudes[0]))

        if projection is not None:
            figure, axes = plt.subplots(figsize=(12, 5), subplot_kw={"projection": projection})
            image = axes.imshow(
                total_map,
                origin="upper",
                extent=extent,
                cmap="magma",
                transform=data_crs,
            )
        else:
            figure, axes = plt.subplots(figsize=(12, 5))
            image = axes.imshow(total_map, origin="upper", extent=extent, cmap="magma")

        axes.set_title("Total input sensitivity")
        axes.set_xlabel("Longitude")
        axes.set_ylabel("Latitude")
        self.maybe_add_coastlines(axes)
        self.add_target_area_patch(axes, data_crs=data_crs)
        self.apply_plot_limits(axes, data_crs=data_crs)
        figure.colorbar(image, ax=axes, shrink=0.8, label="Mean absolute gradient")
        figure.tight_layout()
        figure.savefig(total_path, dpi=150)
        plt.close(figure)

        top_count = min(self.plot_top_k, len(top_channels))
        if top_count <= 0:
            return [total_path]

        top_indices = [channel["index"] for channel in top_channels[:top_count]]
        columns = min(3, top_count)
        rows = math.ceil(top_count / columns)
        if projection is not None:
            figure, axes = plt.subplots(
                rows,
                columns,
                figsize=(5 * columns, 3.5 * rows),
                squeeze=False,
                subplot_kw={"projection": projection},
            )
        else:
            figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)

        for axis in axes.ravel()[top_count:]:
            axis.axis("off")

        for axis, index in zip(axes.ravel(), top_indices):
            if projection is not None:
                image = axis.imshow(
                    np.abs(gradient[index]),
                    origin="upper",
                    extent=extent,
                    cmap="viridis",
                    transform=data_crs,
                )
            else:
                image = axis.imshow(np.abs(gradient[index]), origin="upper", extent=extent, cmap="viridis")
            axis.set_title(self.ordering[index])
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            self.maybe_add_coastlines(axis)
            self.add_target_area_patch(axis, data_crs=data_crs)
            self.apply_plot_limits(axis, data_crs=data_crs)
            figure.colorbar(image, ax=axis, shrink=0.8)

        figure.tight_layout()
        top_path = f"{self.plot_prefix}-top-channels.png"
        figure.savefig(top_path, dpi=150)
        plt.close(figure)

        return [total_path, top_path]

    def model_step(self, model, state):
        if self.rollout_checkpointing and torch.is_grad_enabled():
            return checkpoint(model, state, use_reentrant=False)
        return model(state)

    def write_sensitivity_netcdf(
        self,
        gradients,
        channel_scores,
        total_maps,
        objective_values,
        target_names,
        target_fields,
        target_metrics,
        target_areas,
    ):
        import xarray as xr

        target_count = len(target_names)
        target_area_array = np.full((target_count, 4), np.nan, dtype=np.float32)
        for index, area in enumerate(target_areas):
            if area is not None:
                target_area_array[index, :] = np.asarray(area, dtype=np.float32)

        dataset = xr.Dataset(
            data_vars={
                "sensitivity": (
                    ("target", "channel", "latitude", "longitude"),
                    gradients.astype(np.float32),
                ),
                "channel_score": (
                    ("target", "channel"),
                    channel_scores.astype(np.float32),
                ),
                "total_sensitivity": (
                    ("target", "latitude", "longitude"),
                    total_maps.astype(np.float32),
                ),
                "objective": (
                    ("target",),
                    np.asarray(objective_values, dtype=np.float32),
                ),
                "target_area": (
                    ("target", "area_coord"),
                    target_area_array,
                ),
            },
            coords={
                "target": np.asarray(target_names, dtype="U64"),
                "channel": np.asarray(self.ordering, dtype="U16"),
                "latitude": self.latitudes.astype(np.float32),
                "longitude": self.longitudes.astype(np.float32),
                "area_coord": np.asarray(["north", "west", "south", "east"], dtype="U8"),
                "target_field": ("target", np.asarray(target_fields, dtype="U32")),
                "target_metric": ("target", np.asarray(target_metrics, dtype="U16")),
            },
            attrs={
                "model": "fourcastnetv2-small",
                "lead_time_hours": int(self.lead_time),
                "generated_by": "ai-models-fourcastnetv2 sensitivity mode",
            },
        )

        dataset.to_netcdf(self.sensitivity_path, engine="scipy")

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

    def save_sensitivity(self, gradients, objective_values):
        gradient_maps = np.stack([_channel_maps(gradient) for gradient in gradients], axis=0)
        channel_scores = np.stack([_channel_sensitivity_scores(gradient) for gradient in gradients], axis=0)
        total_maps = np.stack([_total_sensitivity_map(gradient) for gradient in gradients], axis=0)
        peaks = [self.total_sensitivity_peak(total_maps[index]) for index in range(len(self.targets))]

        ranking = np.argsort(channel_scores[0])[::-1]
        top_channels = [
            {
                "index": int(index),
                "channel": self.ordering[index],
                "mean_abs_gradient": float(channel_scores[0, index]),
            }
            for index in ranking[:10]
        ]

        target_names = [target.name for target in self.targets]
        target_fields = [target.field or "" for target in self.targets]
        target_metrics = [target.metric for target in self.targets]
        target_areas = [target.area for target in self.targets]

        self.write_sensitivity_netcdf(
            gradient_maps,
            channel_scores,
            total_maps,
            objective_values,
            target_names,
            target_fields,
            target_metrics,
            target_areas,
        )

        summary = {
            "lead_time_hours": self.lead_time,
            "config_path": self.config_path,
            "targets": [],
        }

        for index, target in enumerate(self.targets):
            ranking = np.argsort(channel_scores[index])[::-1]
            top = [
                {
                    "channel": self.ordering[channel_index],
                    "mean_abs_gradient": float(channel_scores[index, channel_index]),
                }
                for channel_index in ranking[:10]
            ]
            summary["targets"].append(
                {
                    "name": target.name,
                    "field": target.field,
                    "metric": target.metric,
                    "area": None
                    if target.area is None
                    else {
                        "north": target.area[0],
                        "west": target.area[1],
                        "south": target.area[2],
                        "east": target.area[3],
                    },
                    "objective": float(objective_values[index]),
                    "peak_total_sensitivity": peaks[index],
                    "top_channels": top,
                }
            )

        with open(self.summary_path, "w") as file_handle:
            json.dump(summary, file_handle, indent=2)

        LOG.info("Saved sensitivities to %s", self.sensitivity_path)
        LOG.info("Saved sensitivity summary to %s", self.summary_path)

        for index, target in enumerate(self.targets):
            LOG.info(
                "Target %s peak total sensitivity at %.2fN %.2fE",
                target.name,
                peaks[index]["latitude"],
                peaks[index]["longitude"],
            )

        LOG.info("Top sensitivity channels (%s): %s", self.targets[0].name, ", ".join(c["channel"] for c in top_channels[:5]))

        if self.plot_sensitivity:
            base_prefix = self.plot_prefix
            base_target = self.current_target
            for index, target in enumerate(self.targets):
                self.plot_prefix = f"{base_prefix}-{target.name}"
                self.current_target = target
                self.clear_target_caches()
                ranking = np.argsort(channel_scores[index])[::-1]
                target_top = [
                    {
                        "index": int(channel_index),
                        "channel": self.ordering[channel_index],
                        "mean_abs_gradient": float(channel_scores[index, channel_index]),
                    }
                    for channel_index in ranking[:10]
                ]
                plot_paths = self.write_sensitivity_plots(gradient_maps[index], target_top)
                LOG.info("Saved sensitivity plots for %s to %s", target.name, ", ".join(plot_paths))
            self.plot_prefix = base_prefix
            self.current_target = base_target
            self.clear_target_caches()

    def run_forecast(self, model, input_state):
        state = input_state

        with torch.inference_mode():
            with self.stepper(self.hour_steps) as stepper:
                for forecast_index in range(self.lead_time // self.hour_steps):
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
        input_state = input_state.requires_grad_(True)
        state = self.normalise(input_state)
        objective_state = None

        with self.stepper(self.hour_steps) as stepper:
            for forecast_index in range(self.lead_time // self.hour_steps):
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
        for target in self.targets:
            self.current_target = target
            self.clear_target_caches()
            objectives.append(self.sensitivity_objective(objective_state, target))

        gradients = []
        objective_values = []
        for index, objective in enumerate(objectives):
            gradient = torch.autograd.grad(
                objective,
                input_state,
                retain_graph=(index < len(objectives) - 1),
            )[0]
            gradients.append(gradient.detach().cpu().numpy())
            objective_values.append(float(objective.detach().cpu()))

        stacked_gradients = np.stack(gradients, axis=0)
        self.save_sensitivity(stacked_gradients, objective_values)

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
