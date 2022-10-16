import json
import re
import urllib.request
import pandas as pd
import requests
from selenium import webdriver
from Levenshtein import distance, hamming, median
import numpy as np
from selenium.webdriver.support.ui import WebDriverWait
import os
import time
import shutil
import socket
from pypinyin import lazy_pinyin
socket.setdefaulttimeout(300)

class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path

    # 根据不同的apiname生成不同的参数,本示例中未使用全部参数您可在官网(https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html)查看后选择适合业务场景的进行更换
    def gene_params(self, apiname, taskid=None, slice_id=None):
        appid = self.appid
        secret_key = self.secret_key
        upload_file_path = self.upload_file_path
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)
        param_dict = {}

        if apiname == api_prepare:
            # slice_num是指分片数量，如果您使用的音频都是较短音频也可以不分片，直接将slice_num指定为1即可
            slice_num = int(file_len / file_piece_sice) + (0 if (file_len % file_piece_sice == 0) else 1)
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['file_len'] = str(file_len)
            param_dict['file_name'] = file_name
            param_dict['slice_num'] = str(slice_num)
        elif apiname == api_upload:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['slice_id'] = slice_id
        elif apiname == api_merge:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['file_name'] = file_name
        elif apiname == api_get_progress or apiname == api_get_result:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
        return param_dict

    # 请求和结果解析，结果中各个字段的含义可参考：https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html
    def gene_request(self, apiname, data, files=None, headers=None, filename=""):
        response = requests.post(lfasr_host + apiname, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        if result["ok"] == 0:
            print("{} success:".format(apiname) + str(self.upload_file_path))
            return result
        else:
            print("{} error:".format(apiname) + str(result))
            exit(0)
            return result

    # 预处理
    def prepare_request(self):
        return self.gene_request(apiname=api_prepare,
                                 data=self.gene_params(api_prepare))

    # 上传
    def upload_request(self, taskid, upload_file_path):
        file_object = open(upload_file_path, 'rb')
        try:
            index = 1
            sig = SliceIdGenerator()
            while True:
                content = file_object.read(file_piece_sice)
                if not content or len(content) == 0:
                    break
                files = {
                    "filename": self.gene_params(api_upload).get("slice_id"),
                    "content": content
                }
                response = self.gene_request(api_upload,
                                             data=self.gene_params(api_upload, taskid=taskid,
                                                                   slice_id=sig.getNextSliceId()),
                                             files=files)
                if response.get('ok') != 0:
                    # 上传分片失败
                    print('upload slice fail, response: ' + str(response))
                    return False
                print('upload slice ' + str(index) + ' success')
                index += 1
        finally:
            'file index:' + str(file_object.tell())
            file_object.close()
        return True

    # 合并
    def merge_request(self, taskid):
        return self.gene_request(api_merge, data=self.gene_params(api_merge, taskid=taskid))

    # 获取进度
    def get_progress_request(self, taskid):
        return self.gene_request(api_get_progress, data=self.gene_params(api_get_progress, taskid=taskid))

    # 获取结果
    def get_result_request(self, taskid):
        return self.gene_request(api_get_result, data=self.gene_params(api_get_result, taskid=taskid))

    def all_api_request(self,):
        # 1. 预处理
        pre_result = self.prepare_request()
        taskid = pre_result["data"]
        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        # 3 . 文件合并
        self.merge_request(taskid=taskid)
        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    print('task ' + taskid + ' finished')
                    break
                print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        # 5 . 获取结果
        return self.get_result_request(taskid=taskid)


class Musci_info(object):
    def __init__(self, name):
        self.Id = name

    def get_music_info(self):
        url = "https://music.163.com/#/search/m/?s=" + self.Id + "&type=1"
        music_info = []
        driver.get(url)
        time.sleep(5)
        driver.switch_to.frame('contentFrame')
        res = driver.find_element_by_id("m-search")
        res = res.find_element_by_class_name("n-srchrst")

        res_list = res.find_elements_by_class_name("f-cb")
        for i in range(len(res_list)):
            content = res_list[i].find_element_by_class_name('text')
            href = content.find_element_by_tag_name('a').get_attribute('href')
            title = content.find_element_by_tag_name('b').get_attribute('title')
            music_info.append((title, href))
        return music_info

    def get_lyric(self, music_id, music_name):
        url = 'http://music.163.com/api/song/lyric?' + 'id=' + str(music_id) + '&lv=1&kv=1&tv=-1'
        # print('Hi')
        # print(url)
        r = requests.get(url)
        time.sleep(5)
        raw_json = r.text
        if str(r) == '<Response [200]>':
            ch_json = json.loads(raw_json)
            lys_segs = ch_json['lrc']['lyric'].split('\n')
            with open(music_name + '.txt', 'w') as f:
                for seg in lys_segs:
                    f.write(seg + '\n')
            return 1
        else:
            return 0

    def locate_music(self, music_info, Mname):
        for m in music_info:
            if m[0] == Mname:
                return m[1].split('id=')[-1], 1

        rank = np.array(range(len(music_info))) * 2
        sim = [distance(m[0], Mname) for m in music_info]
        loc = np.argmin(sim + rank)
        print(loc, sim + rank)
        if sim[loc] + rank[loc] <= 10:
            return music_info[loc][1].split('id=')[-1], 1
        else:
            return music_info[loc][1].split('id=')[-1], 0

    def locate_music_list(self, music_info, Mname):
        rank = np.array(range(len(music_info))) * 1
        sim = [distance(m[0], Mname) for m in music_info]
        new_list = rank + sim
        if len(new_list) > 3:
            locs = new_list.argsort()[:3]
        else:
            locs = new_list.argsort()[0:]

        return [music_info[_][1].split('id=')[-1] for _ in list(locs)]

    def download_mp3(self, id163, music_name):
        url = apis.track.GetTrackAudio(int(id163))['data'][0]['url']
        try:
            print("正下载：{0}".format(music_name))
            urllib.request.urlretrieve(url, '{0}/{1}.mp3'.format("./data/exp_data/163_audio", music_name))
            print("Finish...")
            time.sleep(3)
        except:
            print("Failed...")
            fail_list.append(music_name)

