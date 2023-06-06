import hashlib
import os
import re
import importlib.util
import inspect

from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.paths import get_ipython_cache_dir
from numba.core.annotations.pretty_annotate import Annotate, AnnotateLLVM


@magics_class
class NumbaMagics(Magics):
    """
    Numba magic commands
    """

    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-ir', '--numbair', action='store_const', const='default',
        dest='annotate_numba_ir', help="Produce a colorized HTML version of the source."
    )
    @magic_arguments.argument(
        '-llvmir', '--llvmir', action='store_const', const='default',
        dest='annotate_llvm_ir', help="Produce a colorized HTML version of the source."
    )
    @cell_magic
    def numba(self, line, cell):
        """Compile the jitted function and display the
        compiled code, control, flow graph, etc.
        """
        args = magic_arguments.parse_argstring(self.numba, line)
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
        if args.annotate_numba_ir:
            # print(python_lines)
            ann = Annotate(
                numba_functions[0],
                python_lines=python_lines,
                python_indent=python_indents
            )
            return ann
        elif args.annotate_llvm_ir:
            fndesc = numba_functions[0].overloads[numba_functions[0].signatures[0]].fndesc
            global_dict = fndesc.global_dict
            python_llvm_ir = global_dict["python_llvm_ir_lines2"]
            llvm_ir_prototype = global_dict["llvm_function_prototype"]
            # print(llvm_ir_prototype)
            # import json

            # pretty = json.dumps(python_llvm_ir, indent=2)
            # print(pretty)
            llvm_ir_lines = self._get_llvm_ir_correct_form(python_llvm_ir, llvm_ir_prototype)["llvm_ir_lines"]
            llvm_ir_indent = self._get_llvm_ir_correct_form(python_llvm_ir, llvm_ir_prototype)["llvm_ir_indent"]
            # print(llvm_ir_lines)
            ann = AnnotateLLVM(
                numba_functions[0],
                python_lines=python_lines,
                python_indent=python_indents,
                llvm_ir_lines=llvm_ir_lines,
                llvm_ir_indent=llvm_ir_indent
            )
            return ann
        return cell

    def _get_llvm_ir_correct_form(self, llvm_ir_dict, llvm_ir_prototype):
        result = {"llvm_ir_lines":{}, "llvm_ir_indent":{}}
        for line in llvm_ir_dict:
            result["llvm_ir_lines"][line] = []
            result["llvm_ir_indent"][line] = []
            for block in llvm_ir_dict[line]:
                if not llvm_ir_dict[line][block]:
                    continue
                else:
                    result["llvm_ir_lines"][line].append((block+":",''))
                    result["llvm_ir_indent"][line].append(0)
                    result["llvm_ir_lines"][line]+=[(x, '') for x in llvm_ir_dict[line][block]]
                    result["llvm_ir_indent"][line]+=[2]*len(llvm_ir_dict[line][block])
        result["llvm_ir_lines"][min(result["llvm_ir_lines"])].insert(0, llvm_ir_prototype)
        result["llvm_ir_indent"][min(result["llvm_ir_indent"])].insert(0, 0)
        result["llvm_ir_lines"][min(result["llvm_ir_lines"])].insert(1, '{')
        result["llvm_ir_indent"][min(result["llvm_ir_indent"])].insert(1, 0)
        result["llvm_ir_lines"][max(result["llvm_ir_lines"])].append('}')
        result["llvm_ir_indent"][max(result["llvm_ir_indent"])].append(0)

        return result

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
        self.numba_file = os.path.join(lib_dir, module_name + '.py')
        with open(self.numba_file, 'w', encoding='utf-8') as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location(module_name, self.numba_file)
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

    def _parse_llvm_lines(self, jitted_function):
        pass

    def _get_llvm_lines(self, jitted_function):
        # code = jitted_function.inspect_llvm()
        pass
