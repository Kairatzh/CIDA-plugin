import pytest
import inspect
from cida_inference.parser_utils import (
    parse_literal_robustly,
    split_arguments_safely,
    bind_arguments_to_signature
)

def test_parse_literal_robustly():
    assert parse_literal_robustly("None") is None
    assert parse_literal_robustly("null") is None
    assert parse_literal_robustly("True") is True
    assert parse_literal_robustly("false") is False
    assert parse_literal_robustly("123") == 123
    assert parse_literal_robustly("'hello'") == "hello"
    assert parse_literal_robustly('"world"') == "world"
    
    # Test JSON and Python dict parsing
    assert parse_literal_robustly('{"a": 1}') == {"a": 1}
    assert parse_literal_robustly("{'a': 1}") == {"a": 1}
    assert parse_literal_robustly("[1, 2, 'three']") == [1, 2, "three"]
    
    # Fallback
    assert parse_literal_robustly("invalid{json") == "invalid{json"

def test_split_arguments_safely():
    # Simple split
    assert split_arguments_safely("1, 2, 3") == ["1", "2", "3"]
    # Split with nested dict/list
    args = split_arguments_safely("method='POST', url='http://localhost', payload={'a': 1, 'b': [2, 3]}")
    assert args == [
        "method='POST'",
        "url='http://localhost'",
        "payload={'a': 1, 'b': [2, 3]}"
    ]
    # Quotes and commas
    assert split_arguments_safely('"a, b", \'c, d\'') == ['"a, b"', "'c, d'"]

def test_bind_arguments_to_signature():
    def dummy_func(a: int, b: str, c: float = 3.14, d: bool = False, **kwargs):
        return a, b, c, d, kwargs
        
    sig = inspect.signature(dummy_func)
    
    # Positionals
    parts = ["10", "'hello'", "2.71", "True"]
    args, kwargs, err = bind_arguments_to_signature(parts, sig)
    assert args == [10, "hello", 2.71, True]
    assert kwargs == {}
    
    # Mixed keywords and defaults
    parts = ["b='world'", "a=5", "extra_param='surprise'"]
    args, kwargs, err = bind_arguments_to_signature(parts, sig)
    assert args == [5, "world", 3.14, False]
    assert kwargs == {"extra_param": "surprise"}
    
    # Nested dict as parameter
    def api_func(payload: dict, headers: dict = None):
        pass
    sig_api = inspect.signature(api_func)
    parts = ["payload={'target': 'idle'}", "headers=None"]
    args, kwargs, err = bind_arguments_to_signature(parts, sig_api)
    assert args == [{"target": "idle"}, None]
