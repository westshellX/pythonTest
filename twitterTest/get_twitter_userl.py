import sys
import os
import traceback
import json
import time
import csv
import requests
from requests.adapters import HTTPAdapter
from twitterscraper import query_tweets_from_user

def validate_config(config):
    # 验证user_id_list
    user_id_list = config['user_id_list']
    if (not isinstance(user_id_list,
               list)) and (not user_id_list.endswith('.txt')):
        sys.exit(u'user_id_list值应为list类型或txt文件路径')
    if not isinstance(user_id_list, list):
        if not os.path.isabs(user_id_list):
            user_id_list = os.path.split(os.path.realpath(__file__))[0] + os.sep + user_id_list
        if not os.path.isfile(user_id_list):
            sys.exit(u'不存在%s文件' % user_id_list)
def get_user_list(file_name):
    """获取文件中的微博id信息"""
    with open(file_name, 'rb') as f:
        lines = f.read().splitlines()
        lines = [line.decode('utf-8') for line in lines]
        user_id_list = [
            line.split(' ')[0] for line in lines
            if len(line.split(' ')) > 0 
        ]
    return user_id_list
def get_filepath(userName, type):
    """获取结果文件路径"""
    try:
        file_dir = os.path.split(
            os.path.realpath(__file__)
        )[0] + os.sep + 'twitter_users_data' + os.sep + userName
        if type == 'img' or type == 'video':
            file_dir = file_dir + os.sep + type
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        if type == 'img' or type == 'video':
            return file_dir
        file_path = file_dir + os.sep + userName + '_' + type
        return file_path
    except Exception as e:
        print('Error: ', e)
        traceback.print_exc()
def download_one_file(userName,fileType,url):
    """下载单个文件(图片/视频)"""
    file_path=get_filepath(userName,fileType)
    try:
        if not os.path.isfile(file_path):
            s = requests.Session()
            s.mount(url, HTTPAdapter(max_retries=5))
            downloaded = s.get(url, cookies='', timeout=(5, 10))
            fileNameStart=url.rfind('/')
            fileName=file_path+os.sep+url[fileNameStart+1:len(url)]
            with open(fileName, 'wb') as f:
                f.write(downloaded.content)
    except Exception as e:
        error_file = file_path + os.sep + 'not_downloaded.txt'
        with open(error_file, 'ab') as f:
            url = userName + ':' + url + '\n'
            f.write(url.encode(sys.stdout.encoding))
        print('Error: ', e)
        traceback.print_exc() 

        
def main():
    try:
        config_path=os.path.split(os.path.realpath(__file__))[0]+os.sep+ 'config.json'
        if not os.path.isfile(config_path) :
            sys.exit(u'当前路径：%s 不存在配置文件config.json' %(os.path.split(os.path.realpath(__file__))[0] + os.sep))
        with open(config_path) as f:
            config = json.loads(f.read())
        validate_config(config)
        
        user_id_list = config['user_id_list']
        if not isinstance(user_id_list, list):
            if not os.path.isabs(user_id_list):
                    user_id_list = os.path.split(os.path.realpath(__file__))[0] + os.sep + user_id_list
            user_id_list =get_user_list(user_id_list)
        for user in user_id_list:
            print(user)
            list_of_tweets=query_tweets_from_user(user,10)
            outPutFileName=get_filepath(user,'data')+'.csv'
            with open(outPutFileName,"w",encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow([
                            "video_url", "reply_to_users"
                        ])
                for t in list_of_tweets:
                    writer.writerow([
                        t.video_url,t.reply_to_users
                    ])
                    for imgUrl in t.img_urls:
                        download_one_file(user,'img',imgUrl)
                    for videoUrl in t.video_url:
                        download_one_file(user,'video',videoUrl)
    except ValueError:
        print('config.json格式不正确')
    except Exception as e:
        print('Error: ',e)
        traceback.print_exe()

if __name__ == '__main__':
    main()

