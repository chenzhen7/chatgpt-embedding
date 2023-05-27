import requests
import re
import os
import tiktoken
import json
import os
import docx
import PyPDF2
import time
import random
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002

max_tokens = 500 # the maximum for text-embedding-ada-002 is 8191
#加载cl100k_base编码，该编码设计用于ada-002模型
tokenizer = tiktoken.get_encoding(embedding_encoding)
apikey = "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"
embedding_url = "https://api.openai-proxy.com/v1/embeddings"

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
        'Connection': 'close',
        'Authorization': 'Bearer ' + apikey,
    }

    # 设置请求体数据
    data = {
        'input': input,
        'model': engine 
    }

   
    for i in range(3):
        
        try:# 发送 POST 请求

            print("post for embeddings")
            response = requests.post(url=embedding_url, headers=headers,data=json.dumps(data),timeout=200)
            # print("request_for_embedding 方法出错 \n response:" + response.text[:200])
            resp_json = response.json()
            time.sleep(5)
            break
        except Exception as e:
            print("request_for_embedding 方法出错 \n response:" + response.text)
            print(e)
            time.sleep(5)
            continue

    return resp_json

# def random_proxy():
#     proxy_list = [
#         '39.99.54.91:80',
#         '183.247.221.119:30001',
#         '114.231.42.156:8888',
#         '182.139.111.228:9000',
#         '121.207.92.141:8888',
#         '183.236.232.160:8080'
#     ]
#     # 从代理 IP 列表中随机选择一个代理
#     proxy_ip = random.choice(proxy_list)
#     # 构建代理的字典格式
#     proxies = {
#         'http': f'http://{proxy_ip}',
#         'https': f'http://{proxy_ip}'
#     }
#     return proxies




def request_for_danvinci003(prompt,temperature,max_tokens,top_p,frequency_penalty,presence_penalty,stop,model):
    # 设置请求头部信息
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apikey,
    }

    data = {
        'prompt' : prompt,
        'temperature' : temperature,
        'max_tokens' : max_tokens,
        'top_p' : top_p,
        'frequency_penalty' : frequency_penalty,
        'presence_penalty' : presence_penalty,
        'stop' : stop,
        'model': model 
    }

    # for i in range(3):
    # 发送 POST 请求
    print("post for https://api.openai-proxy.com/v1/completions")
    response = requests.post('https://api.openai-proxy.com/v1/completions', headers=headers, data=json.dumps(data),timeout=100)

    return response.json()

def request_for_ChatCompletion(messages,model='gpt-3.5-turbo',temperature=0,max_tokens=2048):
    # 设置请求头部信息
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apikey
    }

    data = {
        'messages' : messages,
        'model' : model,
        'temperature' : temperature,
        'max_tokens' : max_tokens
    }
# 发送 POST 请求
    print("post for https://api.openai-proxy.com/v1/chat/completions")
    response = requests.post('https://api.openai-proxy.com/v1/chat/completions', headers=headers, data=json.dumps(data))
    
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
    

def read_text(filepath):
    # 获取文件的绝对路径
    file_path = os.path.abspath(filepath)

    if filepath.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                full_text.append(page_text)
            return '\n'.join(full_text)
    elif filepath.endswith('.docx'):
        # 打开文档
        doc = docx.Document(file_path)
        # 获取文档中所有段落的文本
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)

        return '\n'.join(full_text)
        
    elif filepath.endswith('.txt'):
        # TXT 文档的读取方法
        with open(file_path, 'r',encoding="UTF-8") as txt_file:
            file_text = txt_file.read()
            return file_text

    else:
        return 'Unsupported file format.'