#这段代码是分割句子，最新的嵌入模型可以处理多达 8191 个输入标记的输入，
# 因此大多数行不需要任何分块，但码块会将过长的行拆分为较小的块。
#并生成矢量数据
import requests
import re
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import json

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


#加载cl100k_base编码，该编码设计用于ada-002模型
tokenizer = tiktoken.get_encoding(embedding_encoding)

#读取名为 scraped.csv 的 CSV 文件，其中 index_col=0 参数指定第一列为索引列。该文件将 DataFrame 对象 df 中。
# df = pd.read_csv('processed/scraped.csv', index_col=0)
# #将 DataFrame 对象 df 的列名称更改为 title 和 text。这样做是为了使 DataFrame 的列名称更加明确，以便进行后续的数据分析和处理。
# df.columns = ['title', 'text']

# #对 DataFrame 对象 df 的 text 列进行处理，并将处理结果保存到一个新的 n_tokens 
# #这里使用了一个匿名函数 lambda，将 text 列的每个元素（即每个文件的内容）作为输入，并将该元素传递给 tokenizer.encode() 函数，
# # 该函数将输入的文本转换为 标记（tokens），然后计算标记数并将其赋值给 n_tokens 列。因此，n_tokens 列中的每个元素都表示相应文件的标记数。
# df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# #使用直方图可视化每行标记数的分布
# df.n_tokens.hist()
# plt.show()


max_tokens = 300

import re

#中文的句子分割法
def split_chinese(text):
    # 先将文章按照标点符号分割成段落
    paragraphs = re.split('。|！|？|；|……|………', text)

    # 对于每个段落，将其中的句子分开
    sentences = []
    for p in paragraphs:
        # 忽略空段落
        if not p:
            continue
        
        # 句子的分割依据：逗号、分号、冒号、括号、引号等
        # 如果句子的长度大于10个汉字，则认为是有效句子
        s = re.split('，|、|；|：|（|）|《|》|“|”|‘|’|！|？|。', p)
        # s = [i.strip() for i in s if len(i.strip()) > 10]
        sentences.extend(s)
    
    return sentences

#请求我的代理获得embeding
def request_for_embedding(input,engine='text-embedding-ada-002'):

    # 设置请求头部信息
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + "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"
    }

    # 设置请求体数据
    data = {
        'input': input,
        'model': engine 
    }

    # 发送 POST 请求
    response = requests.post('https://api.openai-proxy.com/v1/embeddings', headers=headers, data=json.dumps(data))
    
    return response.json()



#函数将文本分割为最大数量的标记块
def split_into_many(text, max_tokens = max_tokens):

    #把文章分成句子，这里使用的是英文的分割方法，我们改用中文的分割法
    # sentences = text.split('. ')

    #中文的分割法
    sentences = split_chinese(text=text)

    #计算每个句子的代币数
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    #在元组中循环连接在一起的句子和标记
    for sentence, token in zip(sentences, n_tokens):

        # #如果到目前为止标记的数量加上当前句子中的标记的数量大于最大令牌数量，将区块添加到区块列表中并重置
        # #块和令牌到目前为止
        # if tokens_so_far + token > max_tokens:
        #     chunks.append(". ".join(chunk) + ".")
        #     chunk = []
        #     tokens_so_far = 0

        # #如果当前句子中的标记数大于最大标记数
        # #代币，转到下一句
        # if token > max_tokens:
        #     continue

        # #否则，将句子添加到块中，并将令牌数量添加到总数中
        # chunk.append(sentence)
        # tokens_so_far += token + 1
    
    # 如果该句子的标记数大于最大标记数，将其添加到上一个区块中
        if token > max_tokens:
            if len(chunk) > 0:
                chunks.append(".".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0
            chunks.append(sentence)
        else:
            # 如果到目前为止标记的数量加上当前句子中的标记的数量大于最大令牌数量，将区块添加到区块列表中并重置
            # 块和令牌到目前为止
            if tokens_so_far + token > max_tokens:
                chunks.append(".".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # 将句子添加到块中，并将令牌数量添加到总数中
            chunk.append(sentence)
            tokens_so_far += token + 1

    # 将最后一个块添加到块列表中
    if len(chunk) > 0:
        chunks.append(".".join(chunk) + ".")
    

    return chunks
    
# #缩短的文本列表
# shortened = []

# #循环遍历数dataframe
# for row in df.iterrows():

#     #如果文本为None，则转到下一行
#     if row[1]['text'] is None:
#         continue

#     #如果标记的数量大于最大标记的数量，则使用split_into_many函数将文本分成多个较小的文本块，并将这些文本块添加到shortened列表中
#     if row[1]['n_tokens'] > max_tokens:
#         shortened += split_into_many(row[1]['text'])
    
#     #否则，将文本添加到缩短文本列表
#     else:
#         shortened.append( row[1]['text'] )

#再次可视化更新后的直方图有助于确认行是否已成功拆分为缩短的部分

# print(df.head())


#将嵌入转换为 NumPy 数组是第一步，考虑到在 NumPy 
#数组上运行的许多可用函数，这将在如何使用它方面提供更大的灵活性。它还会将维度展平为一维，这是许多后续操作所需的格式。
import numpy as np
from openai.embeddings_utils import distances_from_embeddings
import requests
import pandas as pd
import time
import json
import pandas as pd
import numpy as np
from flask import Flask,jsonify,request
from flask import Flask, request, jsonify,send_file
import os
from flask_cors import CORS
from pathlib import Path
from utils.embedding_utils import request_for_danvinci003
from file_to_scraped import file_add_embedding,read_text,files_to_embeddings
import traceback


# file_count = 0
# total_size_mb = 0
# total_chars = 0
# for filename in os.listdir('./uploads'):
#     file_path = os.path.join('./uploads', filename)
#     if os.path.isfile(file_path):
#         file_count += 1
#         total_size_mb += round(Path(file_path).stat().st_size / (1024 * 1024), 2)
from utils.embedding_utils import request_for_ChatCompletion

# messages = [{"role": "user", "content": "你好"}]
#         #使用问题和上下文创建一个Completion
# response = request_for_ChatCompletion(
#             messages=messages, 
#         )

# res =  response["choices"][0]["message"]["content"]
# print(res)


# folder_path = './processed'
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(folder_path, file_name)
#         # 读取 CSV 文件并将其添加到 df
#         df_temp = pd.read_csv(file_path)
#         df = pd.concat([df, df_temp], ignore_index=True)
#         # 打印合并后的 DataFrame
# print(df)

import pandas as pd

import pandas as pd
import numpy as np
import re

import pandas as pd
import numpy as np
import re

# def excelToMd(file_path, nan_replacement="-"):
#     # 读取Excel文件
#     df = pd.read_excel(file_path)
    
#     # 删除空的列
#     df = df.dropna(axis=1, how='all')
    
#     # 处理未命名的列名
#     df.columns = ["" if 'Unnamed:' in str(col) else col for col in df.columns]
    
#     # 获取表格的列名和数据
#     columns = df.columns.tolist()
#     data = df.values.tolist()
#     index_column = df.iloc[:, 0].tolist()  # 获取第一列作为行名

#     # 获取表格的数据
#     data = df.iloc[1:, 1:].values.tolist()  # 排除第一行和第一列的数据
    
#     # 构建Markdown表格
#     # markdown_table = ''
#     # markdown_table = '|  | ' + ' | '.join(columns) + ' |\n'
#     # markdown_table += '| --- ' + '| --- ' * len(columns) + ' |\n'
#     formatted_row_data = []
#     for i, row in enumerate(data):
#           # 存储格式化后的每个单元格数据
#         for col, cell in zip(columns, row):
#             if pd.isnull(cell):
#                 formatted_cell = nan_replacement  # 如果单元格为空，使用指定的替代字符串
#             else:
#                 formatted_cell = f'{index_column[i]} {col} {cell}'  # 连接行名、列名和单元格值
#                 formatted_cell = re.sub(r'\s+', ' ', formatted_cell).strip()  # 使用正则表达式进行格式化处理，去除多余的空格
                
#             formatted_row_data.append(formatted_cell)  # 将格式化后的单元格数据添加到列表中
        
#         # formatted_row = ' | '.join(formatted_row_data)  # 将一行的格式化结果通过' | '连接起来
#         # markdown_table += f'| {formatted_row} |\n'  # 将格式化的行添加到Markdown表格中

    
#     return ','.join(formatted_row_data)



import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import re

def excelToMd(file_path, nan_replacement="-"):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 删除空的列
    df = df.dropna(axis=1, how='all')
    
    # 处理未命名的列名
    df.columns = ["" if 'Unnamed:' in str(col) else col for col in df.columns]
    
    # 获取表格的列名和数据
    columns_name = df.columns.tolist()[1:]
    data = df.values.tolist()
    rows_name = df.iloc[:, 0].tolist()  # 获取第一列作为行名
    
    
    # 构建Markdown表格
    # markdown_table = '| ' + ' | '.join(columns) + ' |\n'
    # markdown_table += '| ' + ' | '.join(['---'] * len(columns)) + ' |\n'
    
    def clean_cell(cell):
        # 使用正则表达式替换多个空白字符，并去除首尾空白
        if not pd.isnull(cell):
            return re.sub(r'\s+', ' ', str(cell)).strip()
        else:
            return nan_replacement

    table = []
    for (row,row_name) in zip(data,rows_name):
        # 处理每一行的数据
        row_data = []
        #row[1:]表示每一行的第一个单元格去掉 ，即整张表的第一列不要
        for (col,cell) in zip(columns_name, row[1:]):
            if not pd.isnull(cell):
                #格式化结果
                cleaned_cell = clean_cell(cell)
                # 连接行名、列名和单元格值
                row_name = row_name.strip()
                cleaned_cell = f'{row_name}-{col}-{cleaned_cell}'
                row_data.append(cleaned_cell)
        # 添加到 Markdown 表格中
        # markdown_table += '| ' + ' | '.join(row_data) + ' |\n'
        table += row_data


    
    return '\n'.join(table)


markdown_table = excelToMd("01.xlsx")
# 将Markdown表格写入文件
with open('mk.txt', 'w', encoding='utf-8') as f:
    f.write(markdown_table)

print("Markdown表格已写入到mk.txt文件中。")





def excelToTxt(file_path):
    # 读取Excel文件并填充缺失值
    df = pd.read_excel(file_path)
    df = df.fillna('')  # 将缺失值替换为空字符串

    # 转换为文本格式
    text_data = df.to_string(index=False)

    # 将文本内容写入文件
    with open('excel.txt', 'w', encoding='utf-8') as f:
        f.write(text_data)

    print("Excel表格内容已写入到excel.txt文件中。")


file_path = '01.xlsx'  # 替换为实际的Excel文件路径
excelToTxt(file_path)