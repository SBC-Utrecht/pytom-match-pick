import unittest
import json
from pytom_tm.json import JsonSerializable, CustomJSONEncoder


class TestJSON(unittest.TestCase):
    def test_no_class(self):
        test_dict = {"a": 123}
        with self.assertRaisesRegex(ValueError, "None"):
            _ = JsonSerializable._rebuild_from_dict(test_dict)

    def test_custom_encoder(self):
        class Test1:
            def __init__(self):
                self.test = 12

        class Test2(JsonSerializable):
            def __init__(self):
                self.test = 42

        # Make sure the Encoder raises as expected
        test1 = Test1()
        with self.assertRaisesRegex(TypeError, "Test1"):
            _ = json.dumps(test1, cls=CustomJSONEncoder)

        test2 = Test2()
        out = json.dumps(test2, cls=CustomJSONEncoder)
        self.assertIn("__class__", out)
        self.assertIn("Test2", out)
        self.assertIn(Test2, JsonSerializable._registry.values())
        self.assertIn("42", out)
