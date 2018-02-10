"""
定数用
"""

import set_project


class Const():
    """ valiable setting"""
    word_feat_len = 128
    context_size = 3

    batch_size = 64
    batch_size = 1

    learning_num = 600000
    check_point = 200

    buckets = [(5, 10), (10, 15), (20, 25), (40, 40)]

    seq_len = 3
    """ directory setting"""
    project_dir = set_project.get_path()

    """ word2vec """
    word2vec_train_file = project_dir + \
        "/aozora_text/files/files_all_rnp.txt"
    # word2vec_train_file = project_dir+"/aozora_text/files/files_all.txt"
    word2vec_wait = project_dir + '/lib/model/text8.model'

    """ seq2seq """
    # seq2seq_wait_save_dir = project_dir+'/nn/wait/'
    seq2seq_wait_save_dir = project_dir + '/nn/wait/'
    seq2seq_train_file = project_dir + \
        "/aozora_text/files/files_all_rnp.txt"
    # seq2seq_train_file = project_dir+"/aozora_text/files/tmp.txt"
