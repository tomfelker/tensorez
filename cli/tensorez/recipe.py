"""Recipe (*.toml) parsing with strict validation.

The recipe is the CLI's only input (see CONTRACT.md §1).  Parsing is strict:
unknown sections or keys are hard errors (they are almost always typos, and
the GUI round-trips files it didn't write), and every value is type- and
range-checked with a message pointing at the offending key.

Relative paths — inputs and the output directory alike — resolve against the
**current working directory**, the ordinary shell rule, so a recipe reads the
same way whether you type its paths at a prompt or the GUI does.  The GUI runs
the CLI with the working directory set to the recipe's own folder, which makes
the common case ("recipe sits with the .SER files it processes") come out as
bare filenames and results landing right there.

The recipe says nothing about where run archives or the cache go: those are
machine preferences, not part of the recipe (see cli.py's --runs-dir /
--cache-dir), so a recipe can be moved or shared without dragging one
machine's disk layout along.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RecipeError(ValueError):
    """A problem with the recipe file; message is user-facing."""


# How Bayer sources are converted to channels on read (bayer.py has the
# implementations; the literal is repeated here so recipe parsing stays
# torch-free — keep them in sync):
#   bilinear         full-size 3-channel, missing colors interpolated
#   superpixel_rgb   half-size 3-channel, real photosites only, greens averaged
#   superpixel_rggb  half-size 4-channel (R, G1, G2, B), real photosites only
#   none             keep the mosaic as 1-channel mono (IR-filtered captures)
# Ignored for non-Bayer sources.
DEBAYER_MODES = ("bilinear", "superpixel_rgb", "superpixel_rggb", "none")


@dataclass(frozen=True)
class LightsConfig:
    paths: tuple[str, ...]
    start_frame: int = 0
    frame_step: int = 1
    end_frame: int | None = None
    debayer: str = "bilinear"


@dataclass(frozen=True)
class DarksConfig:
    paths: tuple[str, ...]
    # Same frame selection as [lights] — lets darks be carved out of a capture
    # that contains them (e.g. the empty sky before/after an ISS pass).
    start_frame: int = 0
    frame_step: int = 1
    end_frame: int | None = None
    # Subtract only the master dark's *pattern*, adding its scalar mean back
    # afterwards, so calibrated pixels keep their pedestal instead of
    # scattering around zero.  Useful when the "darks" are really sky frames.
    keep_level: bool = False


@dataclass(frozen=True)
class AlignConfig:
    center_of_mass: bool = True
    per_channel: bool = False   # per-channel CoM: atmospheric dispersion correction
    crop: tuple[int, int] | None = None
    crop_align: int = 2
    crop_offsets: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class LuckyScoringConfig:
    """Whole-frame scalar luckiness, one score per frame (cached).

    ``fourier_bandpass`` is the legacy explorations_lucky metric: mean FFT
    magnitude within a wavelength band — high when fine detail survived the
    seeing.  ``image_squared`` is the classic Muller & Buffington (1974)
    sharpness metric: mean of the squared image.
    """

    metric: str = "fourier_bandpass"     # "fourier_bandpass" | "image_squared"
    min_wavelength_pixels: float = 5.0   # fourier_bandpass band edges
    max_wavelength_pixels: float = 50.0


@dataclass(frozen=True)
class LuckyStackConfig:
    """Plain stacks of the luckiest frames — one output per fraction.

    Each fraction f stacks the best ceil(f * N) frames (at least 1);
    1.0 is the mean of everything.  Requires [lucky_scoring].
    """

    top_fractions: tuple[float, ...] = (0.1,)


@dataclass(frozen=True)
class LocalLuckyConfig:
    """Per-pixel lucky stacking (the legacy local_lucky two-pass scheme)."""

    algorithm: str = "frequency_bands"
    noise_wavelength_pixels: float = 2.0
    crossover_wavelength_pixels: float = 35.0
    isoplanatic_patch_pixels: float = 55.0
    channel_crosstalk: float = 0.0
    stdevs_above_mean: float = 2.5
    steepness: float = 3.0


@dataclass(frozen=True)
class OutputConfig:
    """Where the products land.

    ``dir`` defaults to the recipe's filename without its extension, resolved
    like every other path against the working directory: run ``iss_pass.toml``
    from the folder it lives in and its results appear in ``iss_pass/`` right
    beside it, with no output configuration at all and no chance of mistaking
    them for another recipe's.
    """

    dir: str
    debug_frames: int = 10


# Mode counts accepted by torchmfbd: the wavefront expansion must complete a
# full radial degree of the Zernike/KL pyramid, i.e. cumsum(2..n):
# 2, 5, 9, 14, 20, 27, 35, ...
VALID_MFBD_N_MODES = tuple(n * (n + 1) // 2 - 1 for n in range(2, 15))


@dataclass(frozen=True)
class MfbdConfig:
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
    frames: str = "lucky_top"          # "lucky_top" (needs [lucky_scoring]) | "all"
    top_n: int = 12                    # count of luckiest frames…
    top_fraction: float | None = None  # …or a fraction of all frames, not both
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
    lights: LightsConfig
    darks: DarksConfig | None
    align: AlignConfig
    lucky_scoring: LuckyScoringConfig | None
    lucky_stack: LuckyStackConfig | None
    mfbd: MfbdConfig | None
    local_lucky: LocalLuckyConfig | None
    output: OutputConfig
    path: Path = field(compare=False, default=Path("."))

    @property
    def name(self) -> str:
        """The run's name: the recipe file's stem.  Naming a recipe names the
        run — there is no separate `name` key to keep in sync."""
        return self.path.stem

    @property
    def output_dir(self) -> Path:
        return Path(self.output.dir)

    def resolved_dict(self) -> dict[str, Any]:
        """The recipe as a plain dict with defaults filled in and paths
        absolute — this is what goes into `run_start` and the manifest.

        It is a valid recipe file: every key here can be written back out and
        re-parsed, so the GUI can round-trip it.
        """
        d: dict[str, Any] = {
            "lights": {
                "paths": list(self.lights.paths),
                "start_frame": self.lights.start_frame,
                "frame_step": self.lights.frame_step,
                "debayer": self.lights.debayer,
            },
            "align": {
                "center_of_mass": self.align.center_of_mass,
                "per_channel": self.align.per_channel,
                "crop_align": self.align.crop_align,
                "crop_offsets": list(self.align.crop_offsets),
            },
            "output": {"dir": self.output.dir, "debug_frames": self.output.debug_frames},
        }
        if self.lights.end_frame is not None:
            d["lights"]["end_frame"] = self.lights.end_frame
        if self.darks is not None:
            d["darks"] = {
                "paths": list(self.darks.paths),
                "start_frame": self.darks.start_frame,
                "frame_step": self.darks.frame_step,
                "keep_level": self.darks.keep_level,
            }
            if self.darks.end_frame is not None:
                d["darks"]["end_frame"] = self.darks.end_frame
        if self.align.crop is not None:
            d["align"]["crop"] = list(self.align.crop)
        if self.lucky_scoring is not None:
            sc = self.lucky_scoring
            d["lucky_scoring"] = {"metric": sc.metric}
            if sc.metric == "fourier_bandpass":
                d["lucky_scoring"]["min_wavelength_pixels"] = sc.min_wavelength_pixels
                d["lucky_scoring"]["max_wavelength_pixels"] = sc.max_wavelength_pixels
        if self.lucky_stack is not None:
            d["lucky_stack"] = {"top_fractions": list(self.lucky_stack.top_fractions)}
        if self.local_lucky is not None:
            ll = self.local_lucky
            d["local_lucky"] = {
                "algorithm": ll.algorithm,
                "noise_wavelength_pixels": ll.noise_wavelength_pixels,
                "crossover_wavelength_pixels": ll.crossover_wavelength_pixels,
                "isoplanatic_patch_pixels": ll.isoplanatic_patch_pixels,
                "channel_crosstalk": ll.channel_crosstalk,
                "stdevs_above_mean": ll.stdevs_above_mean,
                "steepness": ll.steepness,
            }
        if self.mfbd is not None:
            dv = self.mfbd
            d["mfbd"] = {
                "method": dv.method,
                "frames": dv.frames,
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
            if dv.frames == "lucky_top":
                if dv.top_fraction is not None:
                    d["mfbd"]["top_fraction"] = dv.top_fraction
                else:
                    d["mfbd"]["top_n"] = dv.top_n
            if dv.pixel_size_um is not None:
                d["mfbd"]["pixel_size_um"] = dv.pixel_size_um
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


def _resolve_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Relative paths resolve against the working directory (see module doc)."""
    return tuple(str(Path(p).resolve()) if not Path(p).is_absolute() else p for p in paths)


def load_recipe(path: str | Path) -> Recipe:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise RecipeError(f"recipe file not found: {path}")
    except tomllib.TOMLDecodeError as e:
        raise RecipeError(f"{path}: invalid TOML: {e}")

    known_sections = {
        "lights", "darks", "align",
        "lucky_scoring", "lucky_stack", "mfbd", "local_lucky", "output",
    }
    unknown = set(data) - known_sections
    if unknown:
        raise RecipeError(
            f"unknown section [{sorted(unknown)[0]}] "
            f"(allowed sections: {', '.join(sorted(known_sections))})"
        )
    for name in data:
        if not isinstance(data[name], dict):
            raise RecipeError(f"top-level key '{name}' must be a section (a TOML table)")
    if "lights" not in data:
        raise RecipeError("missing required section [lights]")

    s = _Section("lights", data["lights"])
    debayer = s.get("debayer", str, "bilinear")
    if debayer not in DEBAYER_MODES:
        raise RecipeError(
            f"[lights] debayer: must be one of {', '.join(DEBAYER_MODES)}, got {debayer!r}"
        )
    lights = LightsConfig(
        paths=_resolve_paths(s.get_str_list("paths", required=True)),
        start_frame=s.get_int("start_frame", 0, minimum=0),
        frame_step=s.get_int("frame_step", 1, minimum=1),
        end_frame=s.get_int("end_frame", None, minimum=1),
        debayer=debayer,
    )
    if lights.end_frame is not None and lights.end_frame <= lights.start_frame:
        raise RecipeError("[lights] end_frame: must be greater than start_frame")
    s.check_no_unknown_keys()

    darks = None
    if "darks" in data:
        s = _Section("darks", data["darks"])
        darks = DarksConfig(
            paths=_resolve_paths(s.get_str_list("paths", required=True)),
            start_frame=s.get_int("start_frame", 0, minimum=0),
            frame_step=s.get_int("frame_step", 1, minimum=1),
            end_frame=s.get_int("end_frame", None, minimum=1),
            keep_level=s.get("keep_level", bool, False),
        )
        if darks.end_frame is not None and darks.end_frame <= darks.start_frame:
            raise RecipeError("[darks] end_frame: must be greater than start_frame")
        s.check_no_unknown_keys()

    s = _Section("align", data.get("align", {}))
    align = AlignConfig(
        center_of_mass=s.get("center_of_mass", bool, True),
        per_channel=s.get("per_channel", bool, False),
        crop=s.get_int_pair("crop", None, minimum=1),
        crop_align=s.get_int("crop_align", 2, minimum=1),
        crop_offsets=s.get_int_pair("crop_offsets", (0, 0)),
    )
    if align.per_channel and not align.center_of_mass:
        raise RecipeError("[align] per_channel requires center_of_mass = true")
    s.check_no_unknown_keys()

    lucky_scoring = None
    if "lucky_scoring" in data:
        s = _Section("lucky_scoring", data["lucky_scoring"])
        metric = s.get("metric", str, "fourier_bandpass")
        if metric not in ("fourier_bandpass", "image_squared"):
            raise RecipeError(
                f"[lucky_scoring] metric: must be 'fourier_bandpass' or 'image_squared', "
                f"got {metric!r}"
            )
        lucky_scoring = LuckyScoringConfig(
            metric=metric,
            min_wavelength_pixels=s.get_number("min_wavelength_pixels", 5.0, minimum=0.0),
            max_wavelength_pixels=s.get_number("max_wavelength_pixels", 50.0, minimum=0.0),
        )
        if (metric == "fourier_bandpass"
                and lucky_scoring.min_wavelength_pixels >= lucky_scoring.max_wavelength_pixels):
            raise RecipeError(
                "[lucky_scoring] min_wavelength_pixels must be smaller than max_wavelength_pixels"
            )
        s.check_no_unknown_keys()

    lucky_stack = None
    if "lucky_stack" in data:
        s = _Section("lucky_stack", data["lucky_stack"])
        fractions = s.get("top_fractions", list)
        if fractions is None:
            fractions = [0.1]
        ok = (len(fractions) > 0
              and all(isinstance(f, (int, float)) and not isinstance(f, bool)
                      and 0.0 < f <= 1.0 for f in fractions))
        if not ok:
            raise RecipeError(
                "[lucky_stack] top_fractions: must be a non-empty list of fractions in "
                "(0, 1] — each stacks the best ceil(fraction * N) frames (1.0 = plain mean)"
            )
        lucky_stack = LuckyStackConfig(top_fractions=tuple(float(f) for f in fractions))
        if lucky_scoring is None:
            raise RecipeError(
                "[lucky_stack] requires [lucky_scoring] — the stack is ordered by the "
                "per-frame luckiness scores"
            )
        s.check_no_unknown_keys()

    local_lucky = None
    if "local_lucky" in data:
        s = _Section("local_lucky", data["local_lucky"])
        algorithm = s.get("algorithm", str, "frequency_bands")
        if algorithm != "frequency_bands":
            raise RecipeError(
                f"[local_lucky] algorithm: only 'frequency_bands' is supported in v0, "
                f"got {algorithm!r}"
            )
        local_lucky = LocalLuckyConfig(
            algorithm=algorithm,
            noise_wavelength_pixels=s.get_number("noise_wavelength_pixels", 2.0, minimum=0.0),
            crossover_wavelength_pixels=s.get_number("crossover_wavelength_pixels", 35.0, minimum=0.0),
            isoplanatic_patch_pixels=s.get_number("isoplanatic_patch_pixels", 55.0, minimum=0.0),
            channel_crosstalk=s.get_number("channel_crosstalk", 0.0, minimum=0.0, maximum=1.0),
            stdevs_above_mean=s.get_number("stdevs_above_mean", 2.5),
            steepness=s.get_number("steepness", 3.0, minimum=0.0),
        )
        if local_lucky.noise_wavelength_pixels >= local_lucky.crossover_wavelength_pixels:
            raise RecipeError(
                "[local_lucky] noise_wavelength_pixels must be smaller than "
                "crossover_wavelength_pixels"
            )
        if local_lucky.crossover_wavelength_pixels >= local_lucky.isoplanatic_patch_pixels:
            raise RecipeError(
                "[local_lucky] crossover_wavelength_pixels must be smaller than "
                "isoplanatic_patch_pixels"
            )
        s.check_no_unknown_keys()

    mfbd = None
    if "mfbd" in data:
        s = _Section("mfbd", data["mfbd"])
        method = s.get("method", str, "torchmfbd")
        if method != "torchmfbd":
            raise RecipeError(f"[mfbd] method: only 'torchmfbd' is supported, got {method!r}")
        frames = s.get("frames", str, "lucky_top")
        if frames not in ("lucky_top", "all"):
            raise RecipeError(f"[mfbd] frames: must be 'lucky_top' or 'all', got {frames!r}")
        if frames == "lucky_top" and lucky_scoring is None:
            raise RecipeError(
                "[mfbd] frames = 'lucky_top' requires [lucky_scoring] — "
                "use frames = 'all' to skip scoring"
            )
        psf_model = s.get("psf_model", str, "kl")
        if psf_model not in ("kl", "zernike"):
            raise RecipeError(f"[mfbd] psf_model: must be 'kl' or 'zernike', got {psf_model!r}")
        opt = s.get("optimizer", str, "adam")
        if opt not in ("adam", "lbfgs"):
            raise RecipeError(f"[mfbd] optimizer: must be 'adam' or 'lbfgs', got {opt!r}")
        n_modes = s.get_int("n_modes", 20, minimum=2)
        if n_modes not in VALID_MFBD_N_MODES:
            raise RecipeError(
                f"[mfbd] n_modes: must complete a full radial degree of the "
                f"wavefront basis; allowed values: {list(VALID_MFBD_N_MODES)}, got {n_modes}"
            )
        wavelengths = s.get("wavelengths_nm", list, required=True)
        if (not wavelengths
                or not all(isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0
                           for w in wavelengths)):
            raise RecipeError("[mfbd] wavelengths_nm: must be a non-empty list of positive "
                              "numbers (nm), one per output color channel")
        cutoff = s.get("frequency_cutoff", list)
        if cutoff is None:
            cutoff = [0.2, 0.3]
        ok = (len(cutoff) == 2
              and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in cutoff)
              and 0.0 < cutoff[0] < cutoff[1] <= 1.0)
        if not ok:
            raise RecipeError("[mfbd] frequency_cutoff: must be [low, high] with "
                              "0 < low < high <= 1 (fractions of the diffraction limit)")
        # Frame budget: a count OR a fraction of all frames, not both.
        top_n = s.get_int("top_n", None, minimum=2)
        top_fraction = s.get_number("top_fraction", None, minimum=1e-9, maximum=1.0)
        if top_n is not None and top_fraction is not None:
            raise RecipeError(
                "[mfbd] top_n and top_fraction are mutually exclusive — give the number "
                "of luckiest frames OR the fraction of all frames, not both"
            )
        # Pixel scale: directly, or computed from the camera geometry —
        # exactly one of the two forms.
        pixel_scale = s.get_number("pixel_scale_arcsec", minimum=1e-6)
        focal_length_mm = s.get_number("focal_length_mm", 2800.0, minimum=1.0)
        barlow = s.get_number("barlow", 1.0, minimum=0.1)
        pixel_size_um = s.get_number("pixel_size_um", minimum=1e-3)
        if pixel_scale is not None and pixel_size_um is not None:
            raise RecipeError(
                "[mfbd] pixel_scale_arcsec and pixel_size_um are mutually exclusive — "
                "give the pixel scale directly OR let it be computed from the camera, not both"
            )
        if pixel_scale is None and pixel_size_um is None:
            raise RecipeError(
                "[mfbd] pixel scale is required: either set pixel_scale_arcsec directly, "
                "or set pixel_size_um (camera photosite pitch in microns, with optional "
                "focal_length_mm and barlow) and it will be computed as "
                "206.265 * pixel_size_um / (focal_length_mm * barlow)"
            )
        if pixel_scale is None:
            pixel_scale = 206.265 * pixel_size_um / (focal_length_mm * barlow)

        mfbd = MfbdConfig(
            method=method,
            frames=frames,
            top_n=top_n if top_n is not None else 12,
            top_fraction=top_fraction,
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
        if mfbd.central_obscuration_cm >= mfbd.diameter_cm:
            raise RecipeError("[mfbd] central_obscuration_cm: must be smaller than diameter_cm")
        s.check_no_unknown_keys()

    if local_lucky is None and lucky_stack is None and mfbd is None:
        raise RecipeError(
            "nothing produces an output: enable at least one of [local_lucky], "
            "[lucky_stack], or [mfbd]"
        )

    s = _Section("output", data.get("output", {}))
    # Default: a directory named after the recipe file, in the working
    # directory — which for the intended workflow is the recipe's own folder.
    out_dir = s.get("dir", str, None)
    if out_dir is None:
        out_dir = path.stem
        if out_dir == path.name:  # extensionless recipe: don't collide with it
            out_dir += "_output"
    out_dir = str(Path(out_dir).resolve())
    output = OutputConfig(dir=out_dir, debug_frames=s.get_int("debug_frames", 10, minimum=0))
    s.check_no_unknown_keys()

    return Recipe(
        lights=lights, darks=darks, align=align,
        lucky_scoring=lucky_scoring, lucky_stack=lucky_stack, mfbd=mfbd,
        local_lucky=local_lucky, output=output, path=path.resolve(),
    )
