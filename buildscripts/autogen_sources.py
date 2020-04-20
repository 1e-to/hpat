# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""

| This script can be used to auto-generate SDC source files from common templates

"""

import sys
import inspect
from pathlib import Path

import sdc.sdc_function_templates as templates_module

encoding_info = '# -*- coding: utf-8 -*-'

copyright_header = '''\
# *****************************************************************************
# Copyright (c) 2020, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************
'''

docstring_header = '''\
"""

| This file contains overloads for various extension types auto-generated with autogen_sources.py

"""
'''

arithmetic_binops_symbols = {
    'add': '+',
    'div': '/',
    'sub': '-',
    'mul': '*',
    'truediv': '/',
    'floordiv': '//',
    'mod': '%',
    'pow': '**',
}

comparison_binops_symbols = {
    'lt': '<',
    'gt': '>',
    'le': '<=',
    'ge': '>=',
    'ne': '!=',
    'eq': '==',
}

target_rel_filename = 'sdc/sdc_autogenerated.py'


if __name__ == '__main__':

    sdc_root_path = Path(__file__).absolute().parents[1]
    target_file_path = sdc_root_path.joinpath(target_rel_filename)

    # read templates_module as text and extract import section to be copied into auto-generated target file
    module_text = inspect.getsource(templates_module)
    module_text_lines = module_text.splitlines(keepends=True)

    # extract copyright text from templates file
    copyright_line_numbers = [k for k, line in enumerate(module_text_lines) if '# *****' in line]
    copyright_section_text = ''.join(module_text_lines[copyright_line_numbers[0]: copyright_line_numbers[1] + 1])

    # extract copyright text from templates file - this only works if imports in it
    # are placed contigiously, i.e. at one place and not intermixed with code
    imports_line_numbers = [k for k, line in enumerate(module_text_lines) if 'import ' in line]
    imports_start_line, import_end_line = min(imports_line_numbers), max(imports_line_numbers)
    import_section_text = ''.join(module_text_lines[imports_start_line: import_end_line + 1])

    series_operator_comp_binop = inspect.getsource(templates_module.sdc_pandas_series_operator_comp_binop)
    # read function templates for arithmetic and comparison operators from templates module
    template_series_arithmetic_binop = inspect.getsource(templates_module.sdc_pandas_series_binop)
    template_series_comparison_binop = inspect.getsource(templates_module.sdc_pandas_series_comp_binop)
    template_series_arithmetic_binop_operator = inspect.getsource(templates_module.sdc_pandas_series_operator_binop)
    template_series_comparison_binop_operator = series_operator_comp_binop
    template_str_arr_comparison_binop = inspect.getsource(templates_module.sdc_str_arr_operator_comp_binop)

    exit_status = -1
    try:
        # open the target file for writing and do the main work
        with target_file_path.open('w', newline='') as file:
            file.write(f'{encoding_info}\n')
            file.write(f'{copyright_section_text}\n')
            file.write(f'{docstring_header}\n')
            file.write(import_section_text)

            # certaing modifications are needed to be applied for templates, so
            # verify correctness of produced code manually
            for name in arithmetic_binops_symbols:
                func_text = template_series_arithmetic_binop.replace('binop', name)
                func_text = func_text.replace(' + ', f' {arithmetic_binops_symbols[name]} ')
                func_text = func_text.replace('def ', f"@sdc_overload_method(SeriesType, '{name}')\ndef ", 1)
                file.write(f'\n\n{func_text}')

            for name in comparison_binops_symbols:
                func_text = template_series_comparison_binop.replace('comp_binop', name)
                func_text = func_text.replace(' < ', f' {comparison_binops_symbols[name]} ')
                func_text = func_text.replace('def ', f"@sdc_overload_method(SeriesType, '{name}')\ndef ", 1)
                file.write(f'\n\n{func_text}')

            for name in arithmetic_binops_symbols:
                if name != "div":
                    func_text = template_series_arithmetic_binop_operator.replace('binop', name)
                    func_text = func_text.replace(' + ', f' {arithmetic_binops_symbols[name]} ')
                    func_text = func_text.replace('def ', f'@sdc_overload(operator.{name})\ndef ', 1)
                    file.write(f'\n\n{func_text}')

            for name in comparison_binops_symbols:
                func_text = template_series_comparison_binop_operator.replace('comp_binop', name)
                func_text = func_text.replace(' < ', f' {comparison_binops_symbols[name]} ')
                func_text = func_text.replace('def ', f'@sdc_overload(operator.{name})\ndef ', 1)
                file.write(f'\n\n{func_text}')

            for name in comparison_binops_symbols:
                func_text = template_str_arr_comparison_binop.replace('comp_binop', name)
                func_text = func_text.replace(' < ', f' {comparison_binops_symbols[name]} ')
                if name == 'ne':
                    func_text = func_text.replace('and not', 'or')
                func_text = func_text.replace('def ', f'@sdc_overload(operator.{name})\ndef ', 1)
                file.write(f'\n\n{func_text}')

    except Exception as e:
        print('Exception of type {}: {} \nwhile writing to a file: {}\n'.format(
            type(e).__name__, e, target_file_path), file=sys.stderr)
        exit_status = 1
    else:
        exit_status = 0

    if not exit_status:
        print('Auto-generation sctipt completed successfully')
    else:
        print('Auto-generation failed, exit_status: {}'.format(exit_status))
    sys.exit(exit_status)
