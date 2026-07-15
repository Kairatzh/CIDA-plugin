import re
import json
import ast
import inspect

def parse_literal_robustly(val_str: str):
    """
    Parses a string representation of a Python or JSON literal safely and robustly.
    Handles single quotes, double quotes, booleans, null/none values, lists, and dicts.
    """
    if not val_str:
        return None
    
    val_str = val_str.strip()
    
    # Handle common constants
    if val_str.lower() in ("none", "null"):
        return None
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
        
    # Remove outer enclosing quotes if it's a simple string
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        inner = val_str[1:-1]
        # If it's a dictionary or list, don't strip outer quotes if they wrap the whole object
        if not (inner.startswith('{') or inner.startswith('[')):
            return inner
            
    # Try standard json loads
    try:
        return json.loads(val_str)
    except Exception:
        pass
        
    # Try ast literal eval (handles Python-style dicts/lists with single quotes)
    try:
        return ast.literal_eval(val_str)
    except Exception:
        pass
        
    # Try converting single quotes to double quotes for JSON parsing
    try:
        cleaned = val_str.replace("'", '"')
        return json.loads(cleaned)
    except Exception:
        pass
        
    # Fallback to returning the raw stripped string
    return val_str

def split_arguments_safely(args_str: str) -> list[str]:
    """
    Splits arguments string by comma while respecting nested quotes, parentheses,
    square brackets, and curly braces.
    """
    parts = []
    current = []
    
    # State tracking
    in_double_quote = False
    in_single_quote = False
    escape_char = False
    
    bracket_stack = [] # Stores opening brackets '(', '[', '{'
    bracket_map = {')': '(', ']': '[', '}': '{'}
    opening_brackets = {'(', '[', '{'}
    
    for char in args_str:
        if escape_char:
            current.append(char)
            escape_char = False
            continue
            
        if char == '\\':
            current.append(char)
            escape_char = True
            continue
            
        # Handle quotes
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current.append(char)
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current.append(char)
        # Handle brackets when not inside quotes
        elif not in_double_quote and not in_single_quote:
            if char in opening_brackets:
                bracket_stack.append(char)
                current.append(char)
            elif char in bracket_map:
                if bracket_stack and bracket_stack[-1] == bracket_map[char]:
                    bracket_stack.pop()
                current.append(char)
            elif char == ',' and not bracket_stack:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
            
    if current:
        parts.append("".join(current).strip())
        
    return [p for p in parts if p]

def bind_arguments_to_signature(parts: list[str], sig: inspect.Signature) -> tuple[list, dict, str]:
    """
    Binds a list of raw argument strings to a function's inspect.Signature.
    Handles positional, keyword, optional, variable positional (*args), 
    and variable keyword (**kwargs) arguments with type casting.
    """
    params = list(sig.parameters.values())
    
    args_list = []
    kwargs_dict = {}
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Find if it is a keyword argument (has '=' outside quotes and brackets)
        eq_idx = -1
        in_dq = False
        in_sq = False
        bracket_depth = 0
        escape = False
        
        for i, c in enumerate(part):
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not in_sq:
                in_dq = not in_dq
            elif c == "'" and not in_dq:
                in_sq = not in_sq
            elif not in_dq and not in_sq:
                if c in ('(', '[', '{'):
                    bracket_depth += 1
                elif c in (')', ']', '}'):
                    bracket_depth = max(0, bracket_depth - 1)
                elif c == '=' and bracket_depth == 0:
                    eq_idx = i
                    break
                    
        if eq_idx != -1:
            key = part[:eq_idx].strip()
            val = part[eq_idx+1:].strip()
            # Unquote key if needed
            if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
                key = key[1:-1]
            kwargs_dict[key] = val
        else:
            args_list.append(part)
            
    # Bind parameters
    bound_args = {}
    
    def cast_type(val_raw, param):
        if val_raw is None:
            return None
        # Parse literal first
        val = parse_literal_robustly(val_raw)
        
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            return val
            
        # Safe type-casting
        try:
            if annotation == int:
                return int(val)
            elif annotation == float:
                return float(val)
            elif annotation == bool:
                if isinstance(val, str):
                    return val.lower() in ("true", "1", "yes")
                return bool(val)
            elif annotation == str:
                return str(val)
            elif annotation == dict and isinstance(val, str):
                parsed = parse_literal_robustly(val)
                if isinstance(parsed, dict):
                    return parsed
            elif annotation == list and isinstance(val, str):
                parsed = parse_literal_robustly(val)
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            pass
            
        return val

    # 1. Map positional arguments
    mapped_names = set()
    pos_param_idx = 0
    
    var_positional_param = None
    var_keyword_param = None
    
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            var_positional_param = p
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            var_keyword_param = p
            
    # Bind positional values
    extra_positional = []
    for arg_val in args_list:
        bound_pos = False
        while pos_param_idx < len(params):
            p = params[pos_param_idx]
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                bound_args[p.name] = cast_type(arg_val, p)
                mapped_names.add(p.name)
                pos_param_idx += 1
                bound_pos = True
                break
            elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                break
            else:
                pos_param_idx += 1
                
        if not bound_pos:
            if var_positional_param:
                extra_positional.append(parse_literal_robustly(arg_val))
            else:
                pass

    # 2. Map keyword arguments
    extra_keyword = {}
    for key, val_raw in kwargs_dict.items():
        p = sig.parameters.get(key)
        if p:
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                bound_args[key] = cast_type(val_raw, p)
                mapped_names.add(key)
        else:
            if var_keyword_param:
                extra_keyword[key] = parse_literal_robustly(val_raw)
            else:
                pass
                
    # 3. Fill missing parameters using defaults or None
    error_msg = None
    for p in params:
        if p.name not in bound_args and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            if p.default != inspect.Parameter.empty:
                bound_args[p.name] = p.default
            else:
                if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    bound_args[p.name] = None
                    
    # Construct final argument lists for calling
    final_args = []
    final_kwargs = {}
    
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            final_args.append(bound_args.get(p.name))
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            final_kwargs[p.name] = bound_args.get(p.name)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            final_args.extend(extra_positional)
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            final_kwargs.update(extra_keyword)
            
    return final_args, final_kwargs, error_msg

def find_first_balanced_action(response: str) -> tuple[str, str]:
    """
    Finds the first action call of the form 'Action: tool_name(args)' in a response string,
    respecting nested parentheses.
    Returns (tool_name, args_str) if found, otherwise (None, None).
    """
    # Look for Action: followed by word and opening parenthesis
    match = re.search(r"Action:\s*(\w+)\(", response)
    if not match:
        return None, None
        
    tool_name = match.group(1)
    start_idx = match.end() # Index right after the opening '('
    
    depth = 1
    in_double_quote = False
    in_single_quote = False
    escape = False
    
    args_chars = []
    
    for i in range(start_idx, len(response)):
        char = response[i]
        
        if escape:
            args_chars.append(char)
            escape = False
            continue
            
        if char == '\\':
            args_chars.append(char)
            escape = True
            continue
            
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            args_chars.append(char)
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            args_chars.append(char)
        elif not in_double_quote and not in_single_quote:
            if char == '(':
                depth += 1
                args_chars.append(char)
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return tool_name, "".join(args_chars)
                else:
                    args_chars.append(char)
            else:
                args_chars.append(char)
        else:
            args_chars.append(char)
            
    # If we reached the end but brackets/quotes were not closed, 
    # return the captured arguments anyway. This ensures a Tool Error is thrown,
    # which gracefully triggers Latent Rollback instead of halting the agent.
    return tool_name, "".join(args_chars)
