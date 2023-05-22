import hashlib
import os
import re
import importlib.util
import inspect

from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.paths import get_ipython_cache_dir
from numba.core.annotations.pretty_annotate import Annotate


@magics_class
class NumbaMagics(Magics):
    """
    Numba magic commands
    """

    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-a', '--annotate', action='store_const', const='default',
        dest='annotate', help="Produce a colorized HTML version of the source."
    )
    @cell_magic
    def numba(self, line, cell):
        """Compile the jitted function and display the
        compiled code, control, flow graph, etc.
        """
        args = magic_arguments.parse_argstring(self.numba, line)
        if args.annotate:
            function_names = self._get_function_names(cell)
            cell = self._preprocess_cell(cell)
            code = cell
            spec = self._build_module_spec(line, code)

            # Import the module
            numba_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(numba_module)
            self._import_all(numba_module)

            numba_functions = self._get_jitted_functions(
                numba_module, function_names
            )
            lines = self._get_lines(cell)
            line_numbers = range(1, len(lines) + 1)
            python_indents = self._get_python_indents(lines, line_numbers)
            python_lines = self._get_python_lines(lines, line_numbers)
            # print(python_lines)
            ann = Annotate(
                numba_functions[0],
                python_lines=python_lines,
                python_indent=python_indents
            )
            return ann
        return cell

    def _preprocess_cell(self, cell):
        cell = cell.strip()
        return cell

    def _get_lines(self, cell):
        lines = [s for s in cell.split('\n')]
        return lines

    def _get_python_lines(self, lines, line_numbers):
        # print(lines)
        # print(line_numbers)
        python_lines = list(zip(line_numbers, map(lambda l: l.strip(), lines)))
        # print(python_lines)
        return python_lines

    def _get_python_indents(self, lines, line_numbers):
        spaces = map(lambda s: len(s) - len(s.lstrip()), lines)
        indents = { line:space for (line,space) in zip(line_numbers, spaces) }
        return indents

    def _get_function_names(self, cell):
        return re.findall(r"def (.*)\(", cell)[0]

    def _get_jitted_functions(self, module, function_names):
        members = inspect.getmembers(module)
        functions = [
            member[1] for member in members if hasattr(member[1], "py_func")
        ]
        return functions

    def _build_module_spec(self, line, code):
        lib_dir = os.path.join(get_ipython_cache_dir(), 'numba')
        key = (code, line)
        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        module_name = "_numba_magic_" + hashlib.sha1(
            str(key).encode('utf-8')).hexdigest()
        numba_file = os.path.join(lib_dir, module_name + '.py')
        with open(numba_file, 'w', encoding='utf-8') as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location(module_name, numba_file)
        return spec

    def _import_all(self, module):
        mdict = module.__dict__
        if '__all__' in mdict:
            keys = mdict['__all__']
        else:
            keys = [k for k in mdict if not k.startswith('_')]

        for k in keys:
            try:
                self.shell.push({k: mdict[k]})
            except KeyError:
                msg = "'module' object has no attribute '%s'" % k
                raise AttributeError(msg)
