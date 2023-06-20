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
from utils.embedding_utils import request_for_danvinci003,request_for_ChatCompletion,request_for_embedding
from file_to_scraped import file_add_embedding,read_text,files_to_embeddings
import traceback
import threading
import chardet

lock = threading.Lock()
app = Flask(__name__)
CORS(app)

apikey = "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"


# #请求我的代理获得embeding
# def request_for_embedding(input,engine='text-embedding-ada-002'):

#     # 设置请求头部信息
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': 'Bearer ' + apikey
#     }

#     # 设置请求体数据
#     data = {
#         'input': input,
#         'model': engine 
#     }

#     # 发送 POST 请求
#     print("post for https://api.openai-proxy.com/v1/embeddings")
#     response = requests.post('https://api.openai-proxy.com/v1/embeddings', headers=headers, data=json.dumps(data))
    
#     return response.json()



"""
    通过从数据框架(dataframe)中找到最相似的上下文来为问题创建上下文(prompt)，并且找到上下文是来自哪些文件，把这些文件的名字返回
"""
def create_context(
    question, df, max_len=1800
):
    
    start = time.time()
    #将问题分词
    # question = " ".join([w for w in list(jb.cut(question))])
    # print("question",question)
    #获取问题的embeddings
    try:
        resp_embedding = request_for_embedding(input=question, engine='text-embedding-ada-002')
        q_embeddings = resp_embedding
    except Exception :
        print(f"resp_embedding = {resp_embedding}")

    end = time.time()
    print("获取问题的embeddings时间：",  end - start)

    start = time.time()
    #获取每个embeddings的距离
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    end = time.time()
    print("获取每个embeddings的距离时间：",  end - start)

    context = []
    filenames = []
    cur_len = 0

    #按距离排序，并将文本添加到上下文中，直到上下文太长
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        #添加文本长度到当前长度
        cur_len += row['n_tokens'] + 4

        #如果上下文太长，则中断
        if cur_len > max_len:
            print(f"上下文长度:{cur_len}")
            break
        
        #否则将它添加到正在返回的上下文中
        context.append(row["text"].replace(' ',''))
            # 检查是否存在'filename'列
    
        filenames.append(row["filename"])

        

    #返回上下文
    return "\n\n##\n\n".join(context),filenames


def answer_question(
    context,
    model="gpt-3.5-turbo",
    question="我是否可以在没有人工审核的情况下将模型输出发布到Twitter？",
    max_len=1500,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    根据数据框架文本中最相似的上下文回答一个问题

    """

    
    #如果是调试，打印原始模型响应
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    #gpt3的接口调用
    # try:
    #     #使用问题和上下文创建一个Completion
    #     response = request_for_danvinci003(
    #         prompt=f"根据下面的上下文回答问题，如果问题不能根据上下文回答, 说 \"很抱歉!我不知道\"\n\n上下文: {context}\n\n---\n\n问题: {question}\n回答:",
    #         temperature=0,
    #         max_tokens=2048,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         stop=stop_sequence,
    #         model='text-davinci-003',
    #     )

    #     res =  response["choices"][0]["text"]
    # except Exception:
    #     print(response)

    # gpt 3.5
    try:
        messages = [{"role": "user", "content": f"根据下面的上下文回答问题，如果不能根据上下文回答, 说 \"很抱歉!我不知道\"\n\n上下文: {context}\n\n---\n\n问题: {question}\n回答:"}]
        #使用问题和上下文创建一个Completion
        response = request_for_ChatCompletion(
            messages=messages, 
        )

        res =  response["choices"][0]["message"]["content"]
        print(repr(res))
    except Exception:
        print(response)

    return res 
    
    # try:
        
    #     start = time.time()
    #    #使用问题和上下文创建一个Completion
    #     response = request_for_ChatCompletion(
    #         messages=[{"role":"user","content": f"根据下面的上下文回答问题，如果问题不能根据上下文回答, 说 \"很抱歉!我不知道\"\n\n上下文: {context}\n\n---\n\n问题: {question}\n回答:"}],
    #         model=model,
    #     )
    #     end = time.time()
    #     print("ChatGPT回复时间：",  end - start)
    #     return response["choices"][0]["message"]["content"].strip()
    # except Exception as e:
    #     print(e)
    #     return ""
    


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# print(answer_question(df, question="近五年来学校共选派九批共多少名大学生参加援藏支教工作?", debug=False))

#主要基于上述我们得到的个人领域的向量化文件，然后将question进行接收，并利用answer_question完成回答，简单进行接口的定义：

#创建一个服务，赋值给APP

#指定接口访问的路径，支持什么请求方式get，post
@app.route('/get_answer',methods=['post'])
#json方式传参
def get_ss():
    # 获取开始时间
    start = time.time()

    # df = pd.read_csv('./processed/embeddings.csv', index_col=0)
    # 遍历 processed 文件夹中的所有文件
    folder_path = './processed'
    df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # 读取 CSV 文件并将其添加到 df
            df_temp = pd.read_csv(file_path,index_col=0)
            df_temp['filename'] = os.path.splitext(file_name)[0]
            df = pd.concat([df, df_temp], ignore_index=True)
            # 打印合并后的 DataFrame

    # 这行代码的目的是将 'embeddings' 列中的字符串转换为对应的对象,然后将这些对象转换为 NumPy 数组
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    print(df.head())  # 默认打印前5行


    end = time.time()
    print("读取embeddings.csv并转化为Numpy数组时间：",  end - start)

    # 获取带json串请求的question参数传入的值
    question = request.json.get('question')
    print("question = " + question)
 

    # 获得上下文信息并且找到上下文是来自哪些文件，把这些文件的名字返回
    context,filenames = create_context(
        question,
        df,
        max_len=1500,
    )
    filenames = list(set(filenames))
    
    print("filenames = " + ' '.join(filenames))

    try:
        msg = answer_question(context, question=question,debug=True)
        code = 1000
        decs = '成功'
    except Exception as e:
        traceback.print_exc()
        code = 9000
        msg = None
        decs = 'openai服务返回异常'

    data = {
        'decs': decs,
        'code': code,
        'msg': msg,
        'filenames':filenames
    }
    print("data",data)

    return jsonify(data)



@app.route('/student/get_answer',methods=['post'])
#json方式传参
def get_ss_student():
    # 获取开始时间
    start = time.time()

    # df = pd.read_csv('./processed/embeddings.csv', index_col=0)
    # 遍历 processed 文件夹中的所有文件
    folder_path = './processed'
    df = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            # 读取 CSV 文件并将其添加到 df
            df_temp = pd.read_csv(file_path,index_col=0)
            df_temp['filename'] = os.path.splitext(file_name)[0]
            df = pd.concat([df, df_temp], ignore_index=True)
            # 打印合并后的 DataFrame

    # df = pd.read_csv('./processed/embeddings.csv', index_col=0)
    # 遍历 processed 文件夹中的所有文件
    studentId = request.json.get('studentId')
    # 遍历./processed/xxx文件夹下的CSV文件
    # 遍历./processed/abc文件夹下的CSV文件（如果文件夹存在）
    subfolder_path = os.path.join(folder_path, studentId)
    if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(subfolder_path, file_name)
                # 读取 CSV 文件并将其添加到 df
                df_temp = pd.read_csv(file_path, index_col=0)
                df_temp['filename'] = os.path.splitext(file_name)[0]
                df = pd.concat([df, df_temp], ignore_index=True)
    else:
        print(f"文件夹 {subfolder_path} 不存在")

    
    # 这行代码的目的是将 'embeddings' 列中的字符串转换为对应的对象,然后将这些对象转换为 NumPy 数组
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


    end = time.time()
    print("读取embeddings.csv并转化为Numpy数组时间：",  end - start)

    # 获取带json串请求的question参数传入的值
    question = request.json.get('question')
    print("question = " + question)
    # 判断请求传入的参数是否在字典里
    context,filenames = create_context(
        question,
        df,
        max_len=1500,
    )
    filenames = list(set(filenames))
    print("filenames = " + ' '.join(filenames))

    try:
        msg = answer_question(context, question=question,debug=True)
        code = 1000
        decs = '成功'
    except Exception as e:
        traceback.print_exc()
        code = 9000
        msg = None
        decs = 'openai服务返回异常'
    data = {
        'decs': decs,
        'code': code,
        'msg': msg,
        'filenames':filenames
    }
    print("data",data)
    return jsonify(data)




UPLOAD_FOLDER = 'uploads'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx','xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/upload', methods=['POST'])
def upload_file():


    if request.method == 'POST':
        # 检查是否提交了文件
        # 线程初始化
    
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        filenames = []

        start = time.time()
        # 处理每个文件
        filename = file.filename
        

        if file and allowed_file(filename):     
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filenames.append(filename)

            embedding_folder_admin = './processed'
            upload_folder_admin =  app.config['UPLOAD_FOLDER']

            for i in range(3):
                try:
                    file_add_embedding(filename=filename,embedding_folder=embedding_folder_admin,upload_folder=upload_folder_admin)
                    break
                except Exception as e:
                    traceback.print_exc()
                    print("\n-----------file_add_embedding出错--------------")
                    time.sleep(5)

                    continue
            threading.Thread(target=save_file_stats).start()

        end = time.time()
        spend = round(end - start,3)

        #线程释放
        return jsonify({'spend': spend})

@app.route('/student/upload', methods=['POST'])
def upload_file_student():

    if request.method == 'POST':
        # 检查是否提交了文件
        # 线程初始化
    
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        studentId = request.values.get('studentId')
        
        filenames = []

        start = time.time()
        # 处理每个文件
        filename = file.filename
  

        upload_folder_student =  os.path.join(app.config['UPLOAD_FOLDER'],studentId)
        if not os.path.exists(upload_folder_student):
            os.makedirs(upload_folder_student)

        if file and allowed_file(filename):
            file.save(os.path.join(upload_folder_student,filename))
            filenames.append(filename)

            embedding_folder_student = os.path.join('./processed',studentId)
         
            for i in range(3):
                try:     
                    file_add_embedding(upload_folder=upload_folder_student,embedding_folder=embedding_folder_student,filename=filename)
                    break
                except Exception as e:
                    traceback.print_exc()
                    print("\n-----------file_add_embedding出错--------------")
                    time.sleep(5)

                    continue
            threading.Thread(target=save_file_stats_student, args=(studentId,)).start()      


        end = time.time()
        spend = round(end - start,3)

        #线程释放
        return jsonify({'spend': spend})


@app.route('/files', methods=['GET'])
def get_file_list():
    file_list = []
    for filename in os.listdir('./uploads'):
        file_path = os.path.join('./uploads', filename)
        if os.path.isfile(file_path):
            file_size_mb = round(Path(file_path).stat().st_size / (1024 * 1024), 2)
            file_list.append({'filename': filename, 'size': round(file_size_mb,3)})

    return jsonify({'files': file_list})


@app.route('/student/files', methods=['POST'])
def get_file_list_student():
    #获得传入的studentId
    studentId = request.json.get('studentId')
    file_list = []
    folder_path = './uploads'
    student_path = os.path.join(folder_path, str(studentId))
    if os.path.exists(student_path) and os.path.isdir(student_path):
        for filename in os.listdir(student_path):
            file_path = os.path.join(student_path, filename)
            if os.path.isfile(file_path):
                file_size_mb = round(Path(file_path).stat().st_size / (1024 * 1024), 2)
                file_list.append({'filename': filename, 'size': round(file_size_mb,3)})
    
    return jsonify({'files': file_list})


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):

    folder_path = './uploads'
    
    # 在目标文件夹中查找与提供的文件名匹配的文件
    matching_files = [file for file in os.listdir(folder_path) if file.startswith(filename)]
    
    if len(matching_files) > 0:
        # 使用匹配到的第一个文件进行下载
        file_path = os.path.join(folder_path, matching_files[0])
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found", 404

@app.route('/student/<studentId>/download/<filename>', methods=['GET'])
def download_file_student(filename,studentId):

    upload_folder_student = os.path.join('./uploads',studentId)
    if os.path.exists(upload_folder_student) and os.path.isdir(upload_folder_student):
        matching_files = [file for file in os.listdir(upload_folder_student) if file.startswith(filename)]
        if len(matching_files) > 0:
            upload_path_student = os.path.join(upload_folder_student,matching_files[0])
            if os.path.isfile(upload_path_student):
                return send_file(upload_path_student, as_attachment=True)


    folder_path = './uploads'
    
    # 在目标文件夹中查找与提供的文件名匹配的文件
    matching_files = [file for file in os.listdir(folder_path) if file.startswith(filename)]
    
    if len(matching_files) > 0:
        # 使用匹配到的第一个文件进行下载
        file_path = os.path.join(folder_path, matching_files[0])
        return send_file(file_path, as_attachment=True)   



@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):

    start = time.time()

    file_path = os.path.join('./uploads', filename)

    name_without_extension = os.path.splitext(filename)[0]
    processed_path = os.path.join('./processed', name_without_extension + '.csv')

    if os.path.isfile(file_path):
        os.remove(file_path)
        os.remove(processed_path)
        end = time.time()
        spend = round(end - start,3)

        threading.Thread(target=save_file_stats).start()

        return jsonify({'message': 'File {} deleted successfully.'.format(filename),'spend':spend})
    
    else:
        return "File not found", 404


@app.route('/student/<studentId>/delete/<filename>', methods=['DELETE'])
def delete_file_student(filename,studentId):

    start = time.time()
    upload_folder_student = os.path.join('./uploads',studentId)
    upload_path_student = os.path.join(upload_folder_student,filename)

    # file_path = os.path.join('./uploads', filename)

    name_without_extension = os.path.splitext(filename)[0]
    processed_folder_student = os.path.join('./processed',studentId)
    processed_path_student = os.path.join(processed_folder_student, name_without_extension + '.csv')

    if os.path.isfile(upload_path_student):
        os.remove(upload_path_student)
        os.remove(processed_path_student)
        end = time.time()
        spend = round(end - start,3)
        return jsonify({'message': 'File {} deleted successfully.'.format(filename),'spend':spend})
    
    else:
        return "File not found", 404



@app.route('/stats', methods=['GET'])
def get_file_stats():
    with open('./stats.json', 'r') as file:
        stats = json.load(file)

    return stats

#每次调用路由处理函数时，统计信息将被重新计算并保存到文件中。
# 您可以在其他地方读取该文件，以随时获取统计信息，而无需每次重新计算。

def save_file_stats():
    file_count = 0
    total_size_mb = 0
    total_chars = 0
    
    for filename in os.listdir('./uploads'):
        file_path = os.path.join('./uploads', filename)
        if os.path.isfile(file_path):
            file_count += 1
            total_size_mb += round(Path(file_path).stat().st_size / (1024 * 1024), 2)
            total_chars +=  len(read_text('uploads/' + filename))
    
    stats = {
        'embedding_num': file_count,
        'embedding_size': round(total_size_mb, 3),
        'embedding_textNum': total_chars
    }
    
    with open('./stats.json', 'w') as file:
        json.dump(stats, file)


@app.route('/student/<studentId>/stats', methods=['GET'])
def get_file_stats_student(studentId):
    stats = {
        'embedding_num': 0,
        'embedding_size': 0,
        'embedding_textNum': 0
    }
    folder_path = f'./{studentId}'
    file_path = os.path.join(folder_path, 'stats.json')

    if not os.path.exists(file_path):
        return stats

    try:
        with open(file_path, 'r') as file:
            stats = json.load(file)
        return stats
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading stats file: {e}")
        return stats



def save_file_stats_student(studentId):
    file_count = 0
    total_size_mb = 0
    total_chars = 0
    for filename in os.listdir('./uploads'):
        file_path = os.path.join('./uploads', filename)
        if os.path.isfile(file_path):
            file_count += 1
            total_size_mb += round(Path(file_path).stat().st_size / (1024 * 1024), 2)
            total_chars +=  len(read_text('uploads/' + filename))

    upload_folder = os.path.join('./uploads',studentId)

    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            file_count += 1
            total_size_mb += round(Path(file_path).stat().st_size / (1024 * 1024), 2)
            total_chars +=  len(read_text(file_path))
    
    stats = {
        'embedding_num': file_count,
        'embedding_size': round(total_size_mb, 3),
        'embedding_textNum': total_chars
    }

    folder_path = f'./{studentId}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, 'stats.json')

    with open(file_path, 'w') as file:
        json.dump(stats, file)



app.run(host='0.0.0.0',port=8083,debug=True)

