{
    'combination_genetic_algorithm':
        {
            'hint':
                {
                    'rus': 'Генетический алгоритм',
                },
            'blocks': 'Process',
            'gui_name':
                {
                    'rus': 'Генетический алгоритм',
                },
            'in_params':
                {
                    'dataset':
                        {
                            'gui_name':
                                {
                                    'rus': 'Датасет',
                                },
                        },
                    'opt_function':
                        {
                            'gui_name':
                                {
                                    'rus': 'Целевая функция',
                                },
                            'gui_type': 'select',
                            'gui_select_values':
                                {
                                    'rus': ['Определение терминалов', 'Определение лайнхола']
                                },
                            'gui_default_values':
                                {
                                    'rus': 'terminals_definition'
                                },
                            'ds_values':
                                {
                                    'rus': ['terminals_definition', 'lh_definition'],
                                },
                        },
                    'opt_function_value':
                        {
                            'gui_name':
                                {
                                    'rus': 'Экстремум целевой функции',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 0
                                },
                        },

                    'dropdown_block':
                        {
                            'gui_name':
                                {
                                    'rus': 'Параметры целевой функции',
                                },
                            'gui_type': 'dropdown_block',
                            'params':
                                {
                                    'path_input':
                                        {
                                            'name': 'path_input',
                                            'gui_name':
                                                {
                                                    'rus': 'path_input'
                                                },
                                            'gui_type': 'api_fs_folder',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'terminals_definition',
                                                        },
                                                },
                                        },
                                    'path_output':
                                        {
                                            'name': 'path_output',
                                            'gui_name':
                                                {
                                                    'rus': 'path_output'
                                                },
                                            'gui_type': 'api_fs_folder',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'terminals_definition',
                                                        },
                                                },
                                        },
                                        'path_input_lh_definition':
                                        {
                                            'name': 'path_input_lh_definition',
                                            'gui_name':
                                                {
                                                    'rus': 'path_input_lh_definition'
                                                },
                                            'gui_type': 'api_fs_folder',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'lh_definition',
                                                        },
                                                },
                                        },
                                    'path_output_lh_definition':
                                        {
                                            'name': 'path_output_lh_definition',
                                            'gui_name':
                                                {
                                                    'rus': 'path_output_lh_definition'
                                                },
                                            'gui_type': 'api_fs_folder',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'lh_definition',
                                                        },
                                                },
                                        },
                                    'path_to_terminals':
                                        {
                                            'name': 'path_to_terminals',
                                            'gui_name':
                                                {
                                                    'rus': 'path_to_terminals'
                                                },
                                            'gui_type': 'api_fs_folder',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'lh_definition',
                                                        },
                                                },
                                        },
                                    'param_for_CGA_num_generations':
                                        {
                                            'gui_name':
                                                {
                                                    'rus': 'param_for_CGA_num_generations',
                                                },
                                            'gui_type': 'input',
                                            'gui_type_value': 'number',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'terminals_definition',
                                                        },
                                                },
                                        },
                                    'param_for_CGA_num_individuals':
                                        {
                                            'gui_name':
                                                {
                                                    'rus': 'param_for_CGA_num_individuals',
                                                },
                                            'gui_type': 'input',
                                            'gui_type_value': 'number',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'terminals_definition',
                                                        },
                                                },
                                        },
                                    'param_for_CGA_early_stop':
                                        {
                                            'gui_name':
                                                {
                                                    'rus': 'param_for_CGA_early_stop',
                                                },
                                            'gui_type': 'input',
                                            'gui_type_value': 'number',
                                            'gui_visible':
                                                {
                                                    'opt_function':
                                                        {
                                                            1: 'terminals_definition',
                                                        },
                                                },
                                        },

                                },
                        },
                    'num_generations':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество поколений',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 100
                                },

                        },
                    'num_individuals':
                        {
                            'gui_name':
                                {
                                    'rus': 'Количество индивидов',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 50
                                },

                        },
                    'early_stop':
                        {
                            'gui_name':
                                {
                                    'rus': 'Критерий остановки',
                                },
                            'gui_type': 'input',
                            'gui_type_value': 'number',
                            'gui_default_values':
                                {
                                    'rus': 25
                                },
                        },
                },
        },
}
