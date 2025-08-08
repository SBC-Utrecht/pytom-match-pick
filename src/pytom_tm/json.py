from dataclasses import asdict, is_dataclass
import json
from typing import Any, Type


class JsonSerializable:
    _registry: dict[str, Type["JsonSerializable"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        JsonSerializable._registry[cls.__name__] = cls

    @classmethod
    def _rebuild_from_dict(cls, data: dict[str, Any]) -> "JsonSerializable":
        """Rebuild any registered class from a dict."""
        class_name = data.pop("__class__", None)
        if class_name not in cls._registry:
            raise ValueError(f"Unknown class: {class_name}")
        target_cls = cls._registry[class_name]
        return target_cls(**data)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, JsonSerializable):
            if is_dataclass(obj):
                result = asdict(obj)
            else:
                result = obj.__dict__.copy()
            result["__class__"] = obj.__class__.__name__
            return result
        return super().default(obj)


class CustomJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "__class__" in obj:
            return JsonSerializable._rebuild_from_dict(obj)
        return obj
