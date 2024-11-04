import pickle

def read_file(file_path, language):
    """
    reading file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        corpus = [{'zh': line.strip()} for line in lines] if language == 'zh' else [{'en': line.strip()} for line in lines]
    return corpus

def read_parallel_files(zh_file_path, en_file_path):
    """
    integrate corpus
    """
    with open(zh_file_path, 'r', encoding='utf-8') as zh_file, open(en_file_path, 'r', encoding='utf-8') as en_file:
        zh_lines = zh_file.readlines()
        en_lines = en_file.readlines()
        corpus = [{'zh': zh_line.strip(), 'en': en_line.strip()} for zh_line, en_line in zip(zh_lines, en_lines)]
    return corpus

def save_to_pkl(corpus, pkl_file_path):
    """
    serialize to pkl files
    """
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(corpus, pkl_file)

def main(zh_file_path=None, en_file_path=None, pkl_file_path='corpus.pkl'):
    """
    main
    """
    if zh_file_path and en_file_path:
        corpus = read_parallel_files(zh_file_path, en_file_path)
    elif zh_file_path:
        corpus = read_file(zh_file_path, 'zh')
    elif en_file_path:
        corpus = read_file(en_file_path, 'en')
    else:
        raise ValueError("need path")

    # 保存到 pkl 文件
    save_to_pkl(corpus, pkl_file_path)
    print(f"saved to {pkl_file_path}")

if __name__ == "__main__":
    zh_file_path = './UNv1.0.en-zh.zh' 
    en_file_path = './UNv1.0.en-zh.en'  

    main(en_file_path)
    