import ast
import json
import sys
import faulthandler
import platform
import io
import sys
import re
import inspect
from datetime import datetime
import signal
from io import StringIO
from unittest.mock import patch, mock_open
from types import ModuleType
from enum import Enum
from decimal import Decimal
import time

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def convert_to_serializable(obj):
    try:
        if isinstance(obj, dict):
            return {
                convert_to_serializable(k): convert_to_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, bool, str, type(None))):
            return obj
        elif hasattr(obj, "__int__"):
            return int(obj)
        elif hasattr(obj, "__float__"):
            return float(obj)
        else:
            import json
            json.dumps(obj)
            return obj
    except Exception:
        return str(obj)
    
def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout

# class Capturing(list):
#     # Add __init__ to accept max_len
#     def __init__(self, max_len=None):
#         super().__init__() # Initialize the list part
#         self.max_len = max_len
#         self._stdout = None
#         self._stringio = None

#     def __enter__(self):
#         self._stdout = sys.stdout
#         # Use the limited StringIO
#         self._stringio = LimitedStringIO(max_len=self.max_len)
#         sys.stdout = self._stringio
#         # No longer need the close override
#         return self

#     def __exit__(self, *args):
#         if self._stringio:
#             self.append(self._stringio.getvalue()) # Get the (limited) value
#             # Let the original StringIO handle cleanup if needed
#             # del self._stringio might still be useful depending on GC behavior
#             del self._stringio
#         if self._stdout:
#             sys.stdout = self._stdout # Restore stdout


class LimitedStringIO(io.StringIO):
    def __init__(self, initial_value='', newline='\n', max_len=None):
        super().__init__(initial_value, newline)
        self.max_len = max_len
        self.current_len = len(initial_value) # Track current length

    def write(self, s):
        if self.max_len is None: # No limit
            written = super().write(s)
            self.current_len += written
            return written
        if self.current_len >= self.max_len:
            return 0 # Limit reached, pretend nothing was written
        remaining_space = self.max_len - self.current_len
        chars_to_write = min(len(s), remaining_space)
        if chars_to_write > 0:
            # Write only the allowed part using the parent's write method
            written = super().write(s[:chars_to_write])
            self.current_len += written
            return written
        else:
            return 0 # No space left to write anything from s


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def parse_value(value_str):
    value_str = value_str.strip()
    if value_str.startswith('{') and value_str.endswith('}'):
        items = value_str[1:-1].split(',')
        return [parse_value(item) for item in items]
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    else:
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            return value_str

def parse_args_to_dict(a):
    args_dict = {}
    for item in a:
        item = item.strip()
        if not item:
            continue
        match = re.match(r"([\w\[\]]+)\s*=\s*(.+)", item)
        if not match:
            continue
        key, value_str = match.groups()
        key = key.replace('[]', '').strip()
        value = parse_value(value_str)
        args_dict[key] = value
    return args_dict


def grade_call_based(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None:
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            # handle floating point comparisons

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()
    return all_results, {"execution time": total_execution}

def grade_call_based_full(
    code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int
):
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [], [{
            "error": "Compilation failed",
            "error_code": -1,
        }]

    method = get_function(compiled_sol, fn_name)
    if method is None:
        return [], [{
            "error": "Function not found",
            "error_code": -1,
        }]

    processed_inputs = []
    processed_outputs = []
    for inputs, output in zip(all_inputs, all_outputs):
        if isinstance(inputs, str):
            try:
                lines = inputs.split("\n")
                processed_input = [json.loads(line) for line in lines]
            except Exception as e:
                args_dict  = parse_args_to_dict(lines)
                sig = inspect.signature(method)
                processed_input = []
                for param in sig.parameters.values():
                    if param.name == 'self':
                        continue
                    if param.name not in args_dict:
                        processed_input.append(None)
                    processed_input.append(args_dict[param.name])
        else:
            processed_input = inputs
        processed_inputs.append(processed_input)
        if isinstance(output, str):
            try:
                processed_output = json.loads(output)
            except json.JSONDecodeError:
                processed_output = output
        else:
            processed_output = output[0]
        processed_outputs.append(processed_output)

    # all_inputs = [[json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs]
    # all_outputs = [json.loads(output) for output in all_outputs]
    all_inputs = processed_inputs
    all_outputs = processed_outputs

    total_execution = 0
    all_results = []
    all_details = []

    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        result_detail = {
            "inputs": gt_inp,
            "expected": gt_out,
            "output": "",
        }
        signal.alarm(timeout)
        faulthandler.enable()

        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            result_detail["output"] = prediction

            if prediction == gt_out:
                all_results.append(True)
                result_detail["status"] = "Correct"
            else:
                all_results.append(False)
                result_detail.update({
                    "error_code": -2,
                    "error_message": "Wrong Answer"
                })

        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                result_detail.update({
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded"
                })
            else:
                all_results.append(-4)
                result_detail.update({
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error"
                })

        finally:
            signal.alarm(0)
            faulthandler.disable()
            all_details.append(result_detail)

    return all_results, all_details



def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            ## CASE 1: exact match
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            ## CASE 2: element-wise comparision
            ## if there are floating elements
            ## use `decimal` library for good floating point comparision
            ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
            ## note that we should always be able to convert to decimals

            success, decimal_prediction_line = convert_line_to_decimals(
                stripped_prediction_line
            )
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def grade_stdio_full(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    code = clean_if_name(code)
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return [], [{"error": "Compilation failed", "error_code": -1}]

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return [], [{"error": "Function not found", "error_code": -1}]

    all_results = []
    all_details = []
    total_execution_time = 0

    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        result_detail = {
            "inputs": gt_inp,
            "expected": gt_out,
            "output": "",
        }
        signal.alarm(timeout)
        faulthandler.enable()
        
        signal.alarm(timeout)
        expected_len = len(str(gt_out))
        max_len = int(expected_len*2)

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    result_detail.update({
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded"
                    })
                else:
                    all_results.append(-4)
                    result_detail.update({
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error"
                    })
                all_details.append(result_detail)
                continue
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        result_detail["output"] = prediction
        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(False)
            result_detail.update({
                "error_code": -2,
                "error_message": "Wrong answer: mismatched output length"
            })
            all_details.append(result_detail)
            continue

        wrong = False
        for output_line_idx, (
            pred_line, 
            gt_line
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            if pred_line == gt_line:
                continue

            success, decimal_pred_line = convert_line_to_decimals(pred_line)
            success_gt, decimal_gt_line = convert_line_to_decimals(gt_line)

            if not (success and success_gt):
                wrong = True
                result_detail.update({
                    "error_code": -2,
                    "error_message": f"Wrong answer at line {output_line_idx}: can't convert to decimal for comparison"
                })
                break

            if decimal_pred_line != decimal_gt_line:
                wrong = True
                result_detail.update({
                    "error_code": -2,
                    "error_message": f"Wrong answer at line {output_line_idx}: {truncatefn(pred_line)} != {truncatefn(gt_line)}"
                })
                break

        if wrong:
            all_results.append(False)
        else:
            all_results.append(True)
            result_detail["status"] = "Correct"

        all_details.append(result_detail)

    return all_results, all_details

def run_test(sample, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    try:
        in_outs = json.loads(sample["input_output"])
    except ValueError as e:
        raise e
        in_outs = None

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None

        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        assert False, "should not happen: test code is none"
        return in_outs, {"error": "no test code provided"}
    elif test is not None:
        results = []
        sol = import_string
        if debug:
            print(f"loading test code = {datetime.now().time()}")
            print(f"the input_output is:\n {in_outs}")
            print(f"the test code is:\n {test}")
        if which_type == CODE_TYPE.call_based:
            signal.alarm(timeout)
            try:
                results, metadata = grade_call_based_full(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    fn_name=method_name,
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], [{
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }]
            finally:
                signal.alarm(0)
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it

            signal.alarm(timeout)
            try:
                results, metadata = grade_stdio_full(
                    code=test,
                    all_inputs=in_outs["inputs"],
                    all_outputs=in_outs["outputs"],
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], [{
                    "error_code": -4,
                    "error_message": f"Error during testing: {e}",
                }]
            finally:
                signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
