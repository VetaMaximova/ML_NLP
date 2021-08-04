import socket
host_name = socket.gethostname() 

# if host_name == 'w42604':   # Attila's Win10 PC
embedding_data_path = { 'w424604' : {'glove.6b.50d' : 'D:\\glove\\glove.6B.50d.txt',
                            'glove.42b.300d' : 'D:\\glove\\glove.42B.300d.txt',
                            'one_hot_encoding' : '', #These are to be implemented later
                            'elmo' : ''} ,
                        'L429746' : {'glove.6b.50d' : '/hdd/data/glove/glove.6B.50d.txt',
                            'glove.42b.300d' : '/hdd/data/glove/glove.42B.300d.txt',
                            'one_hot_encoding' : '', #These are to be implemented later
                            'elmo' : ''},
                        'Veta' : {'glove.6b.50d' : '/hdd/data/glove/glove.6B.50d.txt',
                            'glove.42b.300d' : '/hdd/data/glove/glove.42B.300d.txt',
                            'one_hot_encoding' : '', #These are to be implemented later
                            'elmo' : ''}
                }
# I am not sure we will use DP other than the one from Allan AI.                 
dependency_parser_path = { 'w424604' : {'allan_ai' : 'd:\\allanai_dependency_parser\\biaffine-dependency-parser-ptb-2018.08.23.tar.gz'} ,
                           'L429746' : {'allan_ai' : '/hdd/data/allanai_dependency_parser//biaffine-dependency-parser-ptb-2018.08.23.tar.gz'},
                           'Veta' : {'allan_ai' : 'C:\\MRC\\baikal_data\\biaffine-dependency-parser-ptb-2018.08.23.tar.gz'}
                }
babi_data_path = { 'w424604' : 'D:\\bAbi\\tasks_1-20_v1-2\\en-valid\\',
                   'L429746' : '/hdd/data/bAbI/tasks_1-20_v1-2/en-valid/',
                   'Veta' : '/hdd/data/bAbI/tasks_1-20_v1-2/en-valid/'
                }
log_dir_path =  { 'w424604' : 'D:\\machine_reasoning',
                   'L429746' : '/home/attila/temp_log',
                   'Veta' : '/home/attila/temp_log'
                }
gpu_id_list =  { 'w424604' : [0,],
                   'L429746' : [1,2,],
                   'Veta' : [1,2,]
                }

embedding_dict = embedding_data_path[host_name]
dependency_parser_dict = dependency_parser_path[host_name]
babi_path = babi_data_path[host_name]
LOG_ROOT = log_dir_path[host_name]
gpu_ids = gpu_id_list[host_name]

