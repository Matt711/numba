import hashlib
import os
import re
import importlib.util
import inspect

from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython.paths import get_ipython_cache_dir
from numba.core.annotations.pretty_annotate import Annotate, Annotate2, AnnotateLLVM, AnnotatePTX, AnnotatePTX2, AnnotateBytecode
from numba.cuda.dispatcher import CUDADispatcher
from functools import cache

@magics_class
class NumbaMagics(Magics):
    """
    Numba magic commands
    """

    @cache
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        '-bc', '--bytecode', action='store_const', const='default',
        dest='annotate_bytecode', help="Produce a colorized HTML version of the source."
    )
    @magic_arguments.argument(
        '-ir', '--numbair', action='store_const', const='default',
        dest='annotate_numba_ir', help="Produce a colorized HTML version of the source."
    )
    @magic_arguments.argument(
        '-llvmir', '--llvmir', action='store_const', const='default',
        dest='annotate_llvm_ir', help="Produce a colorized HTML version of the source."
    )
    @magic_arguments.argument(
        '-ptx', '--ptx', action='store_const', const='default',
        dest='annotate_ptx', help="Produce a colorized HTML version of the source."
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
        elif args.annotate_ptx:
            if type(numba_functions[0]) != CUDADispatcher:
                raise Exception(f"PTX only valid for CUDA kernels of type {CUDADispatcher}")   
            asm = numba_functions[0].inspect_asm()
            ptx = next(iter(asm.items()))[1]
            # print(ptx)
            ptx_lines = self._get_ptx_correct_form(ptx)["ptx_lines"]
            # import json
            # pretty = json.dumps(ptx_lines, indent=2)
            # print(pretty)
            # print(ptx_lines[min(ptx_lines)])
            for i, line in enumerate(ptx_lines[min(ptx_lines)]):
                if "ld.param" in line:
                    s = ptx_lines[min(ptx_lines)][i]
                    ptx_lines[min(ptx_lines)][i] = s.split()[0]+" "+s.split()[1]+" ["+re.findall(r'param_\d+', s)[0]+"]"
            ptx_indent = self._get_ptx_correct_form(ptx)["ptx_indent"]
            ann = AnnotatePTX2(
                numba_functions[0],
                python_lines=python_lines,
                python_indent=python_indents,
                ptx_lines=ptx_lines,
                ptx_indent=ptx_indent
            )
            return ann
        elif args.annotate_bytecode:
            bytecode_lines = self._get_bytecode_correct_form(numba_functions[0])["bytecode_lines"]
            bytecode_indent = self._get_bytecode_correct_form(numba_functions[0])["bytecode_indent"]
            import json
            pretty = json.dumps(bytecode_lines, indent=2)
            pretty2 = json.dumps(bytecode_indent, indent=2)
            # print(pretty)
            # print(pretty2)
            ann = AnnotateBytecode(
                numba_functions[0],
                python_lines=python_lines,
                python_indent=python_indents,
                bytecode_lines=bytecode_lines,
                bytecode_indent=bytecode_indent
            )
            return ann
        ann = Annotate2(
            numba_functions[0],
            python_lines=python_lines,
            python_indent=python_indents
        )
        return ann

    def _get_bytecode_correct_form(self, numba_function):
        import dis
        import io
        import sys
        import re
        output = io.StringIO()
        sys.stdout = output
        dis.dis(numba_function)
        sys.stdout = sys.__stdout__
        bytecode_string = output.getvalue()
        # print(bytecode_string)

        bytecode_lines={}
        bytecode_indent={}
        instructions = []
        for inst in bytecode_string.split('\n'):
            if not inst.strip():
                continue
            if len(inst)-len(inst.lstrip()) < 3:
                inst = inst.strip()
                instructions.append(inst)
                # n=inst.split(' ')[0]
                # print(n)
            else:
                instructions.append(inst)
        # print(instructions)
        # print(len(instructions))
        for inst in instructions:
            n=inst.split(' ')[0]
            if n.isnumeric():
                ln=int(n)
                if not ln in bytecode_lines:
                    bytecode_lines[ln] = [" ".join(inst.split()[1:]).strip()]
                    bytecode_indent[ln] = [12]
                else:
                    bytecode_lines[ln].append(" ".join(inst.split()[1:]).strip())
                    bytecode_indent[ln].append(12)
            else:
                bytecode_lines[ln].append(inst.strip())
                bytecode_indent[ln].append(12)
        sys.stdout = sys.__stdout__
        return {"bytecode_lines":bytecode_lines, "bytecode_indent":bytecode_indent}

    def _get_ptx_correct_form(self, ptx):
        ptx_lines = ptx.split('\n')
        where = [(i, x) for i, x in enumerate(ptx.split('\n')) if ".loc" in x]
        d = {"ptx_lines":{}}
        for i in range(len(where)-1):
            ln = int(where[i][1].split()[2])
            if ln==0:
                continue
            if ln not in d["ptx_lines"]:
                d["ptx_lines"][ln] = []
                cleaned = [x for x in ptx_lines[where[i][0]+1:where[i+1][0]]]
                d["ptx_lines"][ln]+="\n".join(cleaned).split('\n')
            else:
                cleaned = [x for x in ptx_lines[where[i][0]+1:where[i+1][0]]]
                d["ptx_lines"][ln]+="\n".join(cleaned).split('\n')
        dd = {"ptx_lines": {}, "ptx_indent": {}}
        for k in d["ptx_lines"]:
            dd["ptx_lines"][k]=[x.strip() for x in d["ptx_lines"][k] if x]
            dd["ptx_indent"][k]=[4*(len(x)-len(x.lstrip())) for x in d["ptx_lines"][k] if x]
        dd["ptx_lines"][min(dd["ptx_lines"])].insert(0, "{")
        dd["ptx_lines"][max(dd["ptx_lines"])].append(ptx_lines[where[-1][0]+1:][:2][0].strip())
        dd["ptx_lines"][max(dd["ptx_lines"])].append("}")
        dd["ptx_indent"][min(dd["ptx_lines"])].insert(0, 0)
        dd["ptx_indent"][max(dd["ptx_lines"])].append(4)
        dd["ptx_indent"][max(dd["ptx_lines"])].append(0)
        return dd

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
