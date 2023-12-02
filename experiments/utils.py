from typing import Any

SPLITTER = "."


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat_params = {}

    def recurse(params: dict[str, Any], path: list[str]) -> None:
        for key, value in params.items():
            new_path = path + [key]
            if isinstance(value, dict):
                recurse(value, new_path)
                continue
            flat_params[SPLITTER.join(new_path)] = value

    recurse(config, [])
    return flat_params


def unflatten_config(config: dict[str, Any]) -> dict[str, Any]:
    _config = {}
    for key, value in config.items():
        c = _config
        parts = key.split(SPLITTER)
        for part in parts[:-1]:
            c[part] = c.get(part, {})
            c = c[part]
        c[parts[-1]] = value
    return _config


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    if len(configs) == 0:
        return {}
    config = {}
    for c in configs:
        config |= flatten_config(c)
    return unflatten_config(config)
