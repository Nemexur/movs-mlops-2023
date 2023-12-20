from typing import Any, Callable, Optional
from dataclasses import dataclass
from pathlib import Path
import re

import click


class DictParamType(click.types.ParamType):
    """A Click type to represent dictionary as parameters for command."""

    name = "DICT"

    def __init__(self, value_type: Callable = str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value_type = value_type

    def convert(
        self,
        value: str,
        param: Optional[click.core.Parameter],
        ctx: Optional[click.core.Context],
    ) -> Optional[dict[str, Any]]:
        """
        Convert value to an appropriate representation.

        Parameters
        ----------
        value: str
            Value assigned to a parameter.
        param: Optional[click.core.Parameter]
            Parameter with assigned value.
        ctx: Optional[click.core.Context]
            Context for CLI.

        Returns
        -------
        Dict[str, Any]
            Key-Value dictionary of parameters.
        """
        extra_vars = super().convert(value=value, param=param, ctx=ctx)
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9:\_\-\.\+\\\/]+)"
        return (
            {
                param: self._value_type(value)
                for param, value in re.findall(regex, extra_vars, flags=re.I)
            }
            if extra_vars is not None
            else None
        )


@dataclass
class EarlyStopping:
    metric: str | None = None
    patience: int = 200
    direction: str = "max"


@dataclass
class SearchHP:
    run: bool = False
    metric: str | None = None
    storage: str | None = None
    trials: int = 10
    seed: int = 13
    train_best: bool = True
    prune: bool = False


@dataclass
class State:
    exp_name: str = None
    exp_dir: Path | None = None
    seed: int = 13
    debug: bool = False
    use_mlflow: bool = False
    extra_vars: dict[str, Any] | None = None


pass_state = click.make_pass_decorator(State, ensure=True)


def name_option(default: str | None = None) -> Callable:
    """
    Add name option to CLI command.

    Parameters
    ----------
    default: str | None (default = None)
        Experiment name.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def wrapper(f: Callable) -> Callable:
        def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
            state: State = ctx.ensure_object(State)
            state.exp_name = value
            return value

        return click.option(
            "-n",
            "--name",
            type=click.STRING,
            help="Experiment name.",
            callback=callback,
            expose_value=False,
            required=False,
            default=default,
            show_default=True,
        )(f)

    return wrapper


def dir_option(default: str | None = None) -> Callable:
    """
    Add dir option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def wrapper(f: Callable) -> Callable:
        def callback(ctx: click.Context, _: click.core.Parameter, value: Path) -> Any:
            state: State = ctx.ensure_object(State)
            state.exp_dir = value
            return value

        return click.option(
            "-d",
            "--dir",
            type=click.Path(exists=False, path_type=Path),
            help="Experiment directory.",
            callback=callback,
            expose_value=False,
            required=False,
            default=default,
            show_default=True,
        )(f)

    return wrapper


def debug_option(f: Callable) -> Callable:
    """
    Add debug option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.debug = value
        return value

    return click.option(
        "--debug",
        is_flag=True,
        help="Run experiment in debug mode.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def seed_option(f: Callable) -> Callable:
    """
    Add seed option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: int) -> Any:
        state: State = ctx.ensure_object(State)
        state.seed = value
        return value

    return click.option(
        "--seed",
        type=click.INT,
        callback=callback,
        expose_value=False,
        required=False,
        default=13,
        show_default=True,
    )(f)


def no_mlflow_option(f: Callable) -> Callable:
    """
    Add mlflow option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.use_mlflow = not value
        return value

    return click.option(
        "--no-mlflow",
        is_flag=True,
        help="Whether to disable mlflow for the experiment or not.",
        callback=callback,
        default=False,
        expose_value=False,
        required=False,
    )(f)


def extra_vars_option(f: Callable) -> Callable:
    """
    Add extra-vars option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: dict[str, Any]) -> Any:
        state: State = ctx.ensure_object(State)
        state.extra_vars = value
        return value

    return click.option(
        "--extra-vars",
        type=DictParamType(),
        help=(
            "Extra variables to inject to yaml config. "
            "Format: {key_name1}={new_value1},{key_name2}={new_value2},..."
        ),
        callback=callback,
        expose_value=False,
        required=False,
        default=None,
        show_default=True,
    )(f)
