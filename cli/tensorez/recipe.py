"""Recipe (*.toml) parsing with strict validation.

The recipe is the CLI's only input (see CONTRACT.md §1).  Parsing is strict:
unknown sections or keys are hard errors (they are almost always typos, and
the GUI round-trips files it didn't write), and every value is type- and
range-checked with a message pointing at the offending key.

Relative paths resolve against the recipe file's directory.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECIPE_VERSION = 0
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RecipeError(ValueError):
    """A problem with the recipe file; message is user-facing."""


@dataclass(frozen=True)
class LightsConfig:
    paths: tuple[str, ...]
    start_frame: int = 0
    frame_step: int = 1
    end_frame: int | None = None


@dataclass(frozen=True)
class DarksConfig:
    paths: tuple[str, ...]


@dataclass(frozen=True)
class AlignConfig:
    center_of_mass: bool = True
    only_even_shifts: bool = False
    per_channel: bool = False   # per-channel CoM: atmospheric dispersion correction
    crop: tuple[int, int] | None = None
    crop_align: int = 2
    crop_offsets: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class LuckyConfig:
    algorithm: str = "frequency_bands"
    noise_wavelength_pixels: float = 2.0
    crossover_wavelength_pixels: float = 35.0
    isoplanatic_patch_pixels: float = 55.0
    channel_crosstalk: float = 0.0
    selection: str = "sigmoid"
    stdevs_above_mean: float = 2.5
    steepness: float = 3.0


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "output"
    debug_frames: int = 10


# Mode counts accepted by torchmfbd: the wavefront expansion must complete a
# full radial degree of the Zernike/KL pyramid, i.e. cumsum(2..n):
# 2, 5, 9, 14, 20, 27, 35, ...
VALID_DECONV_N_MODES = tuple(n * (n + 1) // 2 - 1 for n in range(2, 15))


@dataclass(frozen=True)
class DeconvConfig:
    """Optional multi-frame blind deconvolution of the luckiest frames.

    Physical units: telescope diameter/obscuration in cm, pixel scale in
    arcsec/pixel, wavelengths in nm (one per output color channel).  They are
    converted to torchmfbd's native units (cm / Angstrom / arcsec) on use.

    Aperture defaults are the author's Celestron CPC 1100 (C11): 279.4 mm
    aperture, 95 mm central obstruction, 2800 mm native focal length.

    The pixel scale is given in exactly one of two ways:

    * ``pixel_scale_arcsec`` directly, or
    * ``pixel_size_um`` (camera photosite pitch) with optional
      ``focal_length_mm`` and ``barlow``, computed as
      ``206.265 * pixel_size_um / (focal_length_mm * barlow)``.

    ``pixel_scale_arcsec`` on this dataclass always holds the resolved
    effective value (and so does the resolved recipe in run_start events).
    """

    method: str = "torchmfbd"
    frames: str = "lucky_top"          # "lucky_top" | "all"
    top_n: int = 12
    diameter_cm: float = 27.94
    central_obscuration_cm: float = 9.5
    pixel_scale_arcsec: float = 0.0    # resolved effective value
    focal_length_mm: float = 2800.0
    barlow: float = 1.0
    pixel_size_um: float | None = None
    wavelengths_nm: tuple[float, ...] = ()
    psf_model: str = "kl"              # "kl" | "zernike"
    n_modes: int = 20
    iterations: int = 100
    optimizer: str = "adam"            # "adam" | "lbfgs"
    lr_obj: float = 0.02
    lr_modes: float = 0.08
    apodization_border: int = 0
    frequency_cutoff: tuple[float, float] = (0.2, 0.3)


@dataclass(frozen=True)
class Recipe:
    version: int
    name: str
    lights: LightsConfig
    darks: DarksConfig | None
    align: AlignConfig
    lucky: LuckyConfig
    deconv: DeconvConfig | None
    output: OutputConfig
    path: Path = field(compare=False, default=Path("."))

    def resolved_dict(self) -> dict[str, Any]:
        """The recipe as a plain dict with defaults filled in and paths
        absolute — this is what goes into `run_start` and the manifest."""
        d: dict[str, Any] = {
            "recipe": {"version": self.version, "name": self.name},
            "lights": {
                "paths": list(self.lights.paths),
                "start_frame": self.lights.start_frame,
                "frame_step": self.lights.frame_step,
            },
            "align": {
                "center_of_mass": self.align.center_of_mass,
                "only_even_shifts": self.align.only_even_shifts,
                "per_channel": self.align.per_channel,
                "crop_align": self.align.crop_align,
                "crop_offsets": list(self.align.crop_offsets),
            },
            "lucky": {
                "algorithm": self.lucky.algorithm,
                "noise_wavelength_pixels": self.lucky.noise_wavelength_pixels,
                "crossover_wavelength_pixels": self.lucky.crossover_wavelength_pixels,
                "isoplanatic_patch_pixels": self.lucky.isoplanatic_patch_pixels,
                "channel_crosstalk": self.lucky.channel_crosstalk,
                "selection": self.lucky.selection,
                "stdevs_above_mean": self.lucky.stdevs_above_mean,
                "steepness": self.lucky.steepness,
            },
            "output": {"dir": self.output.dir, "debug_frames": self.output.debug_frames},
        }
        if self.lights.end_frame is not None:
            d["lights"]["end_frame"] = self.lights.end_frame
        if self.darks is not None:
            d["darks"] = {"paths": list(self.darks.paths)}
        if self.align.crop is not None:
            d["align"]["crop"] = list(self.align.crop)
        if self.deconv is not None:
            dv = self.deconv
            d["deconv"] = {
                "method": dv.method,
                "frames": dv.frames,
                "top_n": dv.top_n,
                "diameter_cm": dv.diameter_cm,
                "central_obscuration_cm": dv.central_obscuration_cm,
                # always the resolved effective value, whether given directly
                # or computed from pixel_size_um / focal_length_mm / barlow
                "pixel_scale_arcsec": dv.pixel_scale_arcsec,
                "focal_length_mm": dv.focal_length_mm,
                "barlow": dv.barlow,
                "wavelengths_nm": list(dv.wavelengths_nm),
                "psf_model": dv.psf_model,
                "n_modes": dv.n_modes,
                "iterations": dv.iterations,
                "optimizer": dv.optimizer,
                "lr_obj": dv.lr_obj,
                "lr_modes": dv.lr_modes,
                "apodization_border": dv.apodization_border,
                "frequency_cutoff": list(dv.frequency_cutoff),
            }
            if dv.pixel_size_um is not None:
                d["deconv"]["pixel_size_um"] = dv.pixel_size_um
        return d


class _Section:
    """One TOML table with strict key accounting."""

    def __init__(self, name: str, data: dict[str, Any]):
        self.name = name
        self.data = data
        self.seen: set[str] = set()

    def _fail(self, key: str, message: str) -> None:
        raise RecipeError(f"[{self.name}] {key}: {message}")

    def get(self, key: str, types: type | tuple[type, ...], default: Any = None,
            required: bool = False) -> Any:
        self.seen.add(key)
        if key not in self.data:
            if required:
                self._fail(key, "is required")
            return default
        value = self.data[key]
        # bool is a subclass of int; don't let `true` sneak into int slots.
        if isinstance(value, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
            self._fail(key, f"expected {_type_names(types)}, got a boolean")
        if not isinstance(value, types):
            self._fail(key, f"expected {_type_names(types)}, got {type(value).__name__}")
        return value

    def get_number(self, key: str, default: float | None = None, minimum: float | None = None,
                   maximum: float | None = None) -> float | None:
        value = self.get(key, (int, float), default)
        if value is None:
            return None
        value = float(value)
        if minimum is not None and value < minimum:
            self._fail(key, f"must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            self._fail(key, f"must be <= {maximum}, got {value}")
        return value

    def get_int(self, key: str, default: int | None = None, minimum: int | None = None) -> int | None:
        value = self.get(key, int, default)
        if value is None:
            return None
        if minimum is not None and value < minimum:
            self._fail(key, f"must be >= {minimum}, got {value}")
        return value

    def get_str_list(self, key: str, required: bool = False) -> tuple[str, ...] | None:
        value = self.get(key, list, required=required)
        if value is None:
            return None
        if not value or not all(isinstance(v, str) for v in value):
            self._fail(key, "must be a non-empty list of strings")
        return tuple(value)

    def get_int_pair(self, key: str, default: tuple[int, int] | None = None,
                     minimum: int | None = None) -> tuple[int, int] | None:
        value = self.get(key, list)
        if value is None:
            return default
        ok = len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        if ok and minimum is not None:
            ok = all(v >= minimum for v in value)
        if not ok:
            self._fail(key, f"must be a list of 2 integers"
                            + (f" >= {minimum}" if minimum is not None else ""))
        return (value[0], value[1])

    def check_no_unknown_keys(self) -> None:
        unknown = set(self.data) - self.seen
        if unknown:
            key = sorted(unknown)[0]
            allowed = ", ".join(sorted(self.seen))
            raise RecipeError(
                f"[{self.name}] unknown key '{key}' (allowed keys: {allowed})"
            )


def _type_names(types: type | tuple[type, ...]) -> str:
    if not isinstance(types, tuple):
        types = (types,)
    return " or ".join(t.__name__ for t in types)


def _resolve_paths(paths: tuple[str, ...], base: Path) -> tuple[str, ...]:
    return tuple(str((base / p).resolve()) if not Path(p).is_absolute() else p for p in paths)


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise RecipeError(f"recipe file not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise RecipeError(f"{path}: invalid TOML: {e}")

    known_sections = {"recipe", "lights", "darks", "align", "lucky", "deconv", "output"}
    unknown = set(data) - known_sections
    if unknown:
        raise RecipeError(
            f"unknown section [{sorted(unknown)[0]}] "
            f"(allowed sections: {', '.join(sorted(known_sections))})"
        )
    for name in data:
        if not isinstance(data[name], dict):
            raise RecipeError(f"top-level key '{name}' must be a section (a TOML table)")
    for required in ("recipe", "lights"):
        if required not in data:
            raise RecipeError(f"missing required section [{required}]")

    base = path.resolve().parent

    s = _Section("recipe", data["recipe"])
    version = s.get_int("version", minimum=0)
    if version is None:
        raise RecipeError("[recipe] version: is required")
    if version != RECIPE_VERSION:
        raise RecipeError(f"[recipe] version: only version {RECIPE_VERSION} is supported, got {version}")
    name = s.get("name", str, required=True)
    if not NAME_RE.match(name):
        raise RecipeError(f"[recipe] name: must match [A-Za-z0-9_-]+, got {name!r}")
    s.check_no_unknown_keys()

    s = _Section("lights", data["lights"])
    lights = LightsConfig(
        paths=_resolve_paths(s.get_str_list("paths", required=True), base),
        start_frame=s.get_int("start_frame", 0, minimum=0),
        frame_step=s.get_int("frame_step", 1, minimum=1),
        end_frame=s.get_int("end_frame", None, minimum=1),
    )
    if lights.end_frame is not None and lights.end_frame <= lights.start_frame:
        raise RecipeError("[lights] end_frame: must be greater than start_frame")
    s.check_no_unknown_keys()

    darks = None
    if "darks" in data:
        s = _Section("darks", data["darks"])
        darks = DarksConfig(paths=_resolve_paths(s.get_str_list("paths", required=True), base))
        s.check_no_unknown_keys()

    s = _Section("align", data.get("align", {}))
    align = AlignConfig(
        center_of_mass=s.get("center_of_mass", bool, True),
        only_even_shifts=s.get("only_even_shifts", bool, False),
        per_channel=s.get("per_channel", bool, False),
        crop=s.get_int_pair("crop", None, minimum=1),
        crop_align=s.get_int("crop_align", 2, minimum=1),
        crop_offsets=s.get_int_pair("crop_offsets", (0, 0)),
    )
    if align.per_channel and not align.center_of_mass:
        raise RecipeError("[align] per_channel requires center_of_mass = true")
    if align.per_channel and align.only_even_shifts:
        raise RecipeError(
            "[align] per_channel cannot be combined with only_even_shifts: independent "
            "per-channel shifts cannot preserve the Bayer phase anyway — demosaic without "
            "the even-shift constraint, or disable per_channel"
        )
    s.check_no_unknown_keys()

    s = _Section("lucky", data.get("lucky", {}))
    algorithm = s.get("algorithm", str, "frequency_bands")
    if algorithm != "frequency_bands":
        raise RecipeError(
            f"[lucky] algorithm: only 'frequency_bands' is supported in v0, got {algorithm!r}"
        )
    selection = s.get("selection", str, "sigmoid")
    if selection != "sigmoid":
        raise RecipeError(
            f"[lucky] selection: only 'sigmoid' is supported in v0, got {selection!r}"
        )
    lucky = LuckyConfig(
        algorithm=algorithm,
        noise_wavelength_pixels=s.get_number("noise_wavelength_pixels", 2.0, minimum=0.0),
        crossover_wavelength_pixels=s.get_number("crossover_wavelength_pixels", 35.0, minimum=0.0),
        isoplanatic_patch_pixels=s.get_number("isoplanatic_patch_pixels", 55.0, minimum=0.0),
        channel_crosstalk=s.get_number("channel_crosstalk", 0.0, minimum=0.0, maximum=1.0),
        selection=selection,
        stdevs_above_mean=s.get_number("stdevs_above_mean", 2.5),
        steepness=s.get_number("steepness", 3.0, minimum=0.0),
    )
    if lucky.noise_wavelength_pixels >= lucky.crossover_wavelength_pixels:
        raise RecipeError(
            "[lucky] noise_wavelength_pixels must be smaller than crossover_wavelength_pixels"
        )
    if lucky.crossover_wavelength_pixels >= lucky.isoplanatic_patch_pixels:
        raise RecipeError(
            "[lucky] crossover_wavelength_pixels must be smaller than isoplanatic_patch_pixels"
        )
    s.check_no_unknown_keys()

    deconv = None
    if "deconv" in data:
        s = _Section("deconv", data["deconv"])
        method = s.get("method", str, "torchmfbd")
        if method != "torchmfbd":
            raise RecipeError(f"[deconv] method: only 'torchmfbd' is supported, got {method!r}")
        frames = s.get("frames", str, "lucky_top")
        if frames not in ("lucky_top", "all"):
            raise RecipeError(f"[deconv] frames: must be 'lucky_top' or 'all', got {frames!r}")
        psf_model = s.get("psf_model", str, "kl")
        if psf_model not in ("kl", "zernike"):
            raise RecipeError(f"[deconv] psf_model: must be 'kl' or 'zernike', got {psf_model!r}")
        opt = s.get("optimizer", str, "adam")
        if opt not in ("adam", "lbfgs"):
            raise RecipeError(f"[deconv] optimizer: must be 'adam' or 'lbfgs', got {opt!r}")
        n_modes = s.get_int("n_modes", 20, minimum=2)
        if n_modes not in VALID_DECONV_N_MODES:
            raise RecipeError(
                f"[deconv] n_modes: must complete a full radial degree of the "
                f"wavefront basis; allowed values: {list(VALID_DECONV_N_MODES)}, got {n_modes}"
            )
        wavelengths = s.get("wavelengths_nm", list, required=True)
        if (not wavelengths
                or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0
                           for w in wavelengths)):
            raise RecipeError("[deconv] wavelengths_nm: must be a non-empty list of positive "
                              "numbers (nm), one per output color channel")
        cutoff = s.get("frequency_cutoff", list)
        if cutoff is None:
            cutoff = [0.2, 0.3]
        ok = (len(cutoff) == 2
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in cutoff)
              and 0.0 < cutoff[0] < cutoff[1] <= 1.0)
        if not ok:
            raise RecipeError("[deconv] frequency_cutoff: must be [low, high] with "
                              "0 < low < high <= 1 (fractions of the diffraction limit)")
        # Pixel scale: directly, or computed from the camera geometry —
        # exactly one of the two forms.
        pixel_scale = s.get_number("pixel_scale_arcsec", minimum=1e-6)
        focal_length_mm = s.get_number("focal_length_mm", 2800.0, minimum=1.0)
        barlow = s.get_number("barlow", 1.0, minimum=0.1)
        pixel_size_um = s.get_number("pixel_size_um", minimum=1e-3)
        if pixel_scale is not None and pixel_size_um is not None:
            raise RecipeError(
                "[deconv] pixel_scale_arcsec and pixel_size_um are mutually exclusive — "
                "give the pixel scale directly OR let it be computed from the camera, not both"
            )
        if pixel_scale is None and pixel_size_um is None:
            raise RecipeError(
                "[deconv] pixel scale is required: either set pixel_scale_arcsec directly, "
                "or set pixel_size_um (camera photosite pitch in microns, with optional "
                "focal_length_mm and barlow) and it will be computed as "
                "206.265 * pixel_size_um / (focal_length_mm * barlow)"
            )
        if pixel_scale is None:
            pixel_scale = 206.265 * pixel_size_um / (focal_length_mm * barlow)

        deconv = DeconvConfig(
            method=method,
            frames=frames,
            top_n=s.get_int("top_n", 12, minimum=2),
            diameter_cm=s.get_number("diameter_cm", 27.94, minimum=0.1),
            central_obscuration_cm=s.get_number("central_obscuration_cm", 9.5, minimum=0.0),
            pixel_scale_arcsec=pixel_scale,
            focal_length_mm=focal_length_mm,
            barlow=barlow,
            pixel_size_um=pixel_size_um,
            wavelengths_nm=tuple(float(w) for w in wavelengths),
            psf_model=psf_model,
            n_modes=n_modes,
            iterations=s.get_int("iterations", 100, minimum=1),
            optimizer=opt,
            lr_obj=s.get_number("lr_obj", 0.02, minimum=0.0),
            lr_modes=s.get_number("lr_modes", 0.08, minimum=0.0),
            apodization_border=s.get_int("apodization_border", 0, minimum=0),
            frequency_cutoff=(float(cutoff[0]), float(cutoff[1])),
        )
        if deconv.central_obscuration_cm >= deconv.diameter_cm:
            raise RecipeError("[deconv] central_obscuration_cm: must be smaller than diameter_cm")
        s.check_no_unknown_keys()

    s = _Section("output", data.get("output", {}))
    out_dir = s.get("dir", str, "output")
    if not Path(out_dir).is_absolute():
        out_dir = str((base / out_dir).resolve())
    output = OutputConfig(dir=out_dir, debug_frames=s.get_int("debug_frames", 10, minimum=0))
    s.check_no_unknown_keys()

    return Recipe(
        version=version, name=name, lights=lights, darks=darks,
        align=align, lucky=lucky, deconv=deconv, output=output, path=path.resolve(),
    )
