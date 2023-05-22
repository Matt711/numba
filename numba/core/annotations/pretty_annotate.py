"""
This module implements code highlighting of numba function annotations.
"""

from warnings import warn
from numba.misc.dump_style import NumbaIRLexer

# warn("The pretty_annotate functionality is experimental and might change API",
#          FutureWarning)

def hllines(code, style):
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter
    except ImportError:
        raise ImportError("please install the 'pygments' package")
    pylex = PythonLexer()
    "Given a code string, return a list of html-highlighted lines"
    hf = HtmlFormatter(noclasses=True, style=style, nowrap=True)
    res = highlight(code, pylex, hf)
    return res.splitlines()


def htlines(code, style):
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        # TerminalFormatter does not support themes, Terminal256 should,
        # but seem to not work.
        from pygments.formatters import TerminalFormatter
    except ImportError:
        raise ImportError("please install the 'pygments' package")
    pylex = PythonLexer()
    "Given a code string, return a list of ANSI-highlighted lines"
    hf = TerminalFormatter(style=style)
    res = highlight(code, pylex, hf)
    return res.splitlines()

def hllines_ir(code, style):
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
    except ImportError:
        raise ImportError("please install the 'pygments' package")
    irlex = NumbaIRLexer()
    "Given a code string, return a list of html-highlighted lines"
    hf = HtmlFormatter(noclasses=True, style=style, nowrap=True)
    res = highlight(code, irlex, hf)
    hl_lines = res.splitlines()
    # j=0
    # batches = []
    # for i in batch_sizes:
    #     batches.append(hl_lines[j:i+j])
    #     j+=i
    return hl_lines

def htlines_ir(code, style):
    try:
        from pygments import highlight
        # TerminalFormatter does not support themes, Terminal256 should,
        # but seem to not work.
        from pygments.formatters import TerminalFormatter
    except ImportError:
        raise ImportError("please install the 'pygments' package")
    irlex = NumbaIRLexer()
    "Given a code string, return a list of ANSI-highlighted lines"
    hf = TerminalFormatter(style=style)
    res = highlight(code, irlex, hf)
    # res = highlight(code, irlex, hf)
    hl_lines = res.splitlines()
    # j=0
    # batches = []
    # for i in batch_sizes:
    #     batches.append(hl_lines[j:i+j])
    #     j+=i
    return hl_lines

def get_ansi_template():
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("please install the 'jinja2' package")
    return Template("""
    {%- for func_key in func_data.keys() -%}
        Function name: \x1b[34m{{func_data[func_key]['funcname']}}\x1b[39;49;00m
        {%- if func_data[func_key]['filename'] -%}
        {{'\n'}}In file: \x1b[34m{{func_data[func_key]['filename'] -}}\x1b[39;49;00m
        {%- endif -%}
        {{'\n'}}With signature: \x1b[34m{{func_key[1]}}\x1b[39;49;00m
        {{- "\n" -}}
        {%- for num, line, hl, hc in func_data[func_key]['pygments_lines'] -%}
                {{-'\n'}}{{ num}}: {{hc-}}
                {%- if func_data[func_key]['ir_lines'][num] -%}
                    {%- for ir_line, ir_line_type in func_data[func_key]['ir_lines'][num] %}
                        {{-'\n'}}--{{- ' '*func_data[func_key]['python_indent'][num]}}
                        {{- ' '*(func_data[func_key]['ir_indent'][num][loop.index0]+4)
                        }}{{ir_line }}\x1b[41m{{ir_line_type-}}\x1b[39;49;00m
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
    {%- endfor -%}
    """)
    return ansi_template

def get_html_template():
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("please install the 'jinja2' package")
    return Template("""
    <html>
    <head>
        <style>

            .annotation_table {
                color: #000000;
                font-family: italic;
                margin: 5px;
                width: 100%;
            }

            /* override JupyterLab style */
            .annotation_table td {
                text-align: left;
                background-color: white; /*Might change: transparent*/
                padding: 1px;
            }

            .annotation_table tbody tr:nth-child(even) {
                background: white;
            }

            .annotation_table code
            {
                background-color: white; /*Might change: transparent*/
                white-space: normal;
                font-size: large;
                font-style: italics;
            }

            /* End override JupyterLab style */

            tr:hover {
                background-color: rgba(92, 200, 249, 0.25);
            }

            td.object_tag summary ,
            td.lifted_tag summary{
                font-weight: bold;
                display: list-item;
            }

            span.lifted_tag {
                color: #00cc33;
            }

            span.object_tag {
                color: #cc3300;
            }


            td.lifted_tag {
                background-color: #cdf7d8;
            }

            td.object_tag {
                background-color: #fef5c8;
            }

            code.ir_code {
                color: white;
                font-style: italic;
            }

            .metadata {
                border-bottom: medium solid blue;
                display: inline-block;
                padding: 5px;
                width: 100%;
            }

            .annotations {
                padding: 5px;
            }

            .hidden {
                display: none;
            }

            .buttons {
                padding: 10px;
                cursor: pointer;
            }
        </style>
    </head>

    <body>
        {% for func_key in func_data.keys() %}
            <div class="metadata">
            Function name: {{func_data[func_key]['funcname']}}<br />
            {% if func_data[func_key]['filename'] %}
                in file: {{func_data[func_key]['filename']|escape}}<br />
            {% endif %}
            with signature: {{func_key[1]|e}}
            </div>
            <div class="annotations">
            <table class="annotation_table tex2jax_ignore">
                {%- for num, line, hl, hc in func_data[func_key]['pygments_lines'] -%}
                    {%- if func_data[func_key]['ir_lines'][num] %}
                        <tr><td style="text-align:left;" class="{{func_data[func_key]['python_tags'][num]}}">
                            <details>
                                <summary>
                                    <code>
                                    {{func_data[func_key]['N'] % num}}:
                                    {{'&nbsp;'*func_data[func_key]['python_indent'][num]}}{{hl}}
                                    </code>
                                </summary>
                                <table class="annotation_table">
                                    <tbody>
                                        {%- for ir_num, ir_lines, ir_hl, ir_hc in func_data[func_key]['pygments_ir_lines_'+num|string] %}
                                            <tr class="ir_code">
                                                <td style="text-align: left;">
                                                    <code>
                                                        {{'&nbsp;'*func_data[func_key]['ir_indent'][num][ir_num-1]}}{{ir_hl}}
                                                    </code>
                                                </td>
                                            </tr>
                                        {%- endfor -%}
                                    </tbody>
                                </table>
                            </details>
                        </td></tr>
                    {% else -%}
                        <tr><td style="text-align:left;" class="{{func_data[func_key]['python_tags'][num]}}">
                            <code>
                                {{func_data[func_key]['N'] % num}}:
                                {{'&nbsp;'*func_data[func_key]['python_indent'][num]}}{{hl}}
                            </code>
                        </td></tr>
                    {%- endif -%}
                {%- endfor -%}
            </table>
            </div>
        {% endfor %}
    </body>
    </html>
    """)


def reform_code(annotation):
    """
    Extract the code from the Numba annotation datastructure. 

    Pygments can only highlight full multi-line strings, the Numba
    annotation is list of single lines, with indentation removed.
    """
    ident_dict = annotation['python_indent']
    s= ''
    for n,l in annotation['python_lines']:
        s = s+' '*ident_dict[n]+l+'\n'
    return s

def reform_ir_code(annotation, i):
    """
    Extract the code from the Numba annotation datastructure. 

    Pygments can only highlight full multi-line strings, the Numba
    annotation is list of single lines, with indentation removed.
    """
    ident_dict = annotation['ir_indent']
    s= ''
    for n,l in annotation[f'ir_lines_{i}']:
        s = s+' '*ident_dict[i][n-1]+l+'\n'
    return s

def seperate_ir_lines(annotation):
    # lines = list(list(ann.items())[0][1]['python_indent'].keys())
    # lines = list(annotation.items())[0][1]['lines']
    ir = list(annotation.items())[0][1]['ir_lines']
    lines = [l for l in ir.keys() if ir[l] != []]
    for i in lines:
        list(annotation.items())[0][1][f'ir_lines_{i}'] = [(j+1, l[0]) for j, l in enumerate(ir[i])]

class Annotate:
    """
    Construct syntax highlighted annotation for a given jitted function:

    Example:

    >>> import numba
    >>> from numba.pretty_annotate import Annotate
    >>> @numba.jit
    ... def test(q):
    ...     res = 0
    ...     for i in range(q):
    ...         res += i
    ...     return res
    ...
    >>> test(10)
    45
    >>> Annotate(test)

    The last line will return an HTML and/or ANSI representation that will be
    displayed accordingly in Jupyter/IPython.

    Function annotations persist across compilation for newly encountered
    type signatures and as a result annotations are shown for all signatures
    by default.

    Annotations for a specific signature can be shown by using the
    ``signature`` parameter.

    >>> @numba.jit
    ... def add(x, y):
    ...     return x + y
    ...
    >>> add(1, 2)
    3
    >>> add(1.3, 5.7)
    7.0
    >>> add.signatures
    [(int64, int64), (float64, float64)]
    >>> Annotate(add, signature=add.signatures[1])  # annotation for (float64, float64)
    """
    def __init__(self, function, signature=None, **kwargs):
        self.function = function
        style = kwargs.get('style', 'default')
        if not function.signatures:
            raise ValueError('function need to be jitted for at least one signature')
        ann = function.get_annotation_info(signature=signature)
        self.ann = ann
        self.python_lines = kwargs.get('python_lines', None)
        self.python_indent = kwargs.get('python_indent', None)
        if self.python_lines:
            list(ann.items())[0][1]['python_lines'] = self.python_lines
        if self.python_indent:
            list(ann.items())[0][1]['python_indent'] = self.python_indent

        for k,v in ann.items():
            res = hllines(reform_code(v), style)
            rest = htlines(reform_code(v), style)
            v['pygments_lines'] = [(a,b,c, d) for (a,b),c, d in zip(v['python_lines'], res, rest)]

        import math
        list(ann.items())[0][1]['N'] = f"%0{int(math.log10(len(list(ann.items())[0][1]['python_lines']))+1)}d"
        seperate_ir_lines(ann)
        ir = list(ann.items())[0][1]['ir_lines']
        lines = [l for l in ir.keys() if ir[l] != []]
        for i in lines:
            for k,v in ann.items():
                res = hllines_ir(reform_ir_code(v, i), style)
                rest = htlines_ir(reform_ir_code(v, i), style)
                v[f'pygments_ir_lines_{i}'] = [(a,b,c,d) for (a,b),c,d in zip(v[f'ir_lines_{i}'], res, rest)] # Batch into pygments_ir_lines_1, 2, ...

    def _repr_html_(self):
        return get_html_template().render(func_data=self.ann)

    def __repr__(self):
        return get_ansi_template().render(func_data=self.ann)
