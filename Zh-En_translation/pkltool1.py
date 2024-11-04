import pickle

def read_file_to_corpus(file_path):
    """
    读取指定路径的 TXT 文件，并将每一行转换成指定的字典结构。
    """
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            corpus.append({str(index): line.strip()})
    return corpus

def save_corpus_to_pkl(corpus, pkl_file_path):
    """
    将语料库序列化为 pkl 文件。
    """
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(corpus, pkl_file)

def save_corpus_to_txt(corpus, txt_file_path):
    """
    将语料库保存为 TXT 文件。
    """
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        for item in corpus:
            for key, value in item.items():
                txt_file.write(f"{key}: {value}\n")

def main(txt_file_path, pkl_file_path='corpus_zh.pkl', output_txt_file_path='data_zh.txt'):
    """
    主函数，读取 TXT 文件并保存为 pkl 文件和 TXT 文件。
    """
    corpus = read_file_to_corpus(txt_file_path)
    save_corpus_to_pkl(corpus, pkl_file_path)
    save_corpus_to_txt(corpus, output_txt_file_path)
    print(f"语料库已保存到 {pkl_file_path} 和 {output_txt_file_path}")

if __name__ == "__main__":
    txt_file_path = './Data/UNv1.0.en-zh.zh' 
    main(txt_file_path)

    