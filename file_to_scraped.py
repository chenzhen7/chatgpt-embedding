#把文本内容转化为csv
import os
import jieba as jb
import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
from utils.embedding_utils import split_into_many,request_for_embedding,read_text
import time
import traceback
import re


# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 500
tokenizer = tiktoken.get_encoding(embedding_encoding)

def remove_newlines(serie):
    # serie = serie.str.replace('\n', ' ')
    # serie = serie.str.replace('\r', ' ')
    # serie = serie.str.replace('\\n', ' ',regex=True)
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie
#将[文件夹]的所有文件转为scraped.csv
def files_to_embeddings(floders_path:str):
    # Create a list to store the text files
    texts=[]

    #获取文本目录中的所有文本文件
    for file in os.listdir(floders_path):
        text = read_text('uploads/' + file)
        text = " ".join([w for w in list(jb.cut(text))])
        texts.append((file, text))

    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    #将文本列设置为删除换行符后的原始文本
    df['text'] = df['fname'] + "\n" + remove_newlines(df.text)
    # df.to_csv('processed/scraped.csv')
    df.columns = ['title', 'text']  
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    #缩短的文本列表
    shortened = []

    #循环遍历数dataframe
    for row in df.iterrows():

        #如果文本为None，则转到下一行
        if row[1]['text'] is None:
            continue

        #如果标记的数量大于最大标记的数量，则使用split_into_many函数将文本分成多个较小的文本块，并将这些文本块添加到shortened列表中
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        
        #否则，将文本添加到缩短文本列表
        else:
            shortened.append( row[1]['text'] )

    #再次可视化更新后的直方图有助于确认行是否已成功拆分为缩短的部分
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    start = time.time()
    df['embeddings'] = df.text.apply(lambda x: request_for_embedding(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    end = time.time()
    spend = end - start
    print(f"生成embeddings花费时间:{spend}")
    df.to_csv('processed/embeddings.csv')



    # df.head()

#将文件转化为embeding后追加到embeding.csv中
def file_add_embedding(upload_folder:str,embedding_folder:str,filename:str):
    # Create a list to store the text files
    #文件路径
    uploadpath = os.path.join(upload_folder,filename)
    texts=[]
    #判断是否为excel表格
    isExcel = filename.endswith('.xlsx')

    text = read_text(uploadpath)
    # 这行代码的目的是将文本 text 进行分词，并将分词后的词语通过空格连接成一个新的字符串。
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\t+', '\t', text)
    text = " ".join([w for w in list(jb.cut(text))])
    texts.append((filename, text))

    # 从文本列表中创建一个数据框。

    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    #将文本列设置为删除换行符后的原始文本
    df['text'] = df['fname'] + "\n" + df.text
    #不写入 直接添加到embedding
    # df.to_csv('processed/scraped.csv')
    tokenizer = tiktoken.get_encoding(embedding_encoding)

    #将 DataFrame 对象 df 的列名称更改为 title 和 text。这样做是为了使 DataFrame 的列名称更加明确，以便进行后续的数据分析和处理。
    df.columns = ['title', 'text']
    #对 DataFrame 对象 df 的 text 列进行处理，并将处理结果保存到一个新的 n_tokens 
    #这里使用了一个匿名函数 lambda，将 text 列的每个元素（即每个文件的内容）作为输入，并将该元素传递给 tokenizer.encode() 函数，
    # 该函数将输入的文本转换为 标记（tokens），然后计算标记数并将其赋值给 n_tokens 列。因此，n_tokens 列中的每个元素都表示相应文件的标记数。
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # 使用直方图可视化每行标记数的分布
    # df.n_tokens.hist()
    # plt.show()

    #缩短的文本列表
    shortened = []
    
    #循环遍历数dataframe
    for row in df.iterrows():

        #如果文本为None，则转到下一行
        if row[1]['text'] is None:
            continue

        #如果标记的数量大于最大标记的数量，则使用split_into_many函数将文本分成多个较小的文本块，并将这些文本块添加到shortened列表中
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'],isExcel=isExcel,filename = filename)
        
        #否则，将文本添加到缩短文本列表
        else:
            shortened.append( row[1]['text'] )
    
    # for i in shortened:
    print(len(shortened))

    #再次可视化更新后的直方图有助于确认行是否已成功拆分为缩短的部分
    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    
    # 使用直方图可视化每行标记数的分布
    # df.n_tokens.hist()
    # plt.show()
    
    start = time.time() 
    # print(df['text'].head())
    
    # df['embeddings'] = df.text.apply(lambda x: request_for_embedding(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    def process_text(x):
        for i in range(3):
            try:
                result = request_for_embedding(input=x, engine='text-embedding-ada-002')
                break
            except Exception as e:
                traceback.print_exc()
                print("\n-----------process_text出错--------------")
                print(f"\n-----------result={result}--------------")

                time.sleep(5)
                continue
                
        # time.sleep(5)  # 等待5秒
        return result

    df['embeddings'] = df.text.apply(process_text)
    
    end = time.time()
    print(f"生成embeddings花费时间:{end -start}")

    # old_df = pd.read_csv('processed/embeddings.csv', index_col=0)
    # # 将新的数据框添加到现有数据框中
    # new_df = pd.concat([old_df, df], ignore_index=True)
    # 只需要获取文件名部分而不包括扩展名
    name_without_extension = os.path.splitext(filename)[0]
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)

    df.to_csv(os.path.join(embedding_folder, name_without_extension + '.csv'),escapechar='\\')
    # print(df.head())



# if __name__ == '__main__':
    

    
