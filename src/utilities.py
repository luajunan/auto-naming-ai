import os
import ast

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
# sys.path.extend([".", ".."])

from pycparser import c_ast, parse_file, c_generator

DATA_DIR = os.path.join(os.getcwd(), "data")


# A simple visitor for FuncDef nodes that prints the names and
# locations of function definitions.
class CFuncVisitor(c_ast.NodeVisitor):

    def __init__(self):
        self.function_bodies = []
        self.generator = c_generator.CGenerator()

    def visit_FuncDef(self, node):
        self.function_bodies.append(self.generator.visit(node))


def parse_c_file(filename):
    # Note that cpp is used. Provide a path to your own cpp or
    # make sure one exists in PATH.
    ast = parse_file(
        filename,
        use_cpp=True,
        cpp_path="gcc",
        cpp_args=["-E", r"-Iutils/fake_libc_include"],
    )

    v = CFuncVisitor()
    v.visit(ast)
    return v.function_bodies


class PyFuncVisitor(ast.NodeVisitor):

    def visit_FunctionDef(self, node):
        print("Function:", node.name)
        self.generic_visit(node)
        # self.function_bodies.append(node)

    def visit_Assign(self, node):
        print("Assignment:", ast.dump(node))
        self.generic_visit(node)
        # self.function_bodies.append(node)

    def visit_Call(self, node):
        print("Function Call:", ast.dump(node))
        self.generic_visit(node)
        # self.function_bodies.append(node)


def extract_py_function_body(source_code, func_node):
    start_line = func_node.lineno
    end_line = func_node.end_lineno
    function_body_lines = source_code.splitlines()[start_line - 1 : end_line]
    return "\n".join(function_body_lines)


def parse_py_file(filename):
    function_bodies = []
    r = open(filename, "r")
    tree = ast.parse(r.read())
    visitor = PyFuncVisitor()
    visitor.visit(tree)

    with open(filename, "r") as f:
        source_code = f.read()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_body = extract_py_function_body(source_code, node)
            print("Function Extracted:", node.name)
            function_bodies.append(function_body)

    return function_bodies


def filter_output(llm_response):
    try:
        json_output = ast.literal_eval(llm_response.strip())
        return json_output
    except:
        print("LLM did not return a JSON format, please try again.")
        return None
