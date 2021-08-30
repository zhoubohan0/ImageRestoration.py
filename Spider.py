import requests as r
from lxml import etree
import parsel
import re
import os
import time


def downloadImage(baseurl,destpath,page = 1):
    start = time.time()
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.67'}

    for i in range(1, 6):
        url = os.path.join(baseurl,'list-{}.html'.format(i))
        # 1.确定地址，发送请求
        response = r.get(url,headers=headers)
        response.encoding=response.apparent_encoding
        # 2.xpath方法
        raw_url = parsel.Selector(response.text).xpath('//*[@id="waterfall_auto_box"]/div/dl/dd/a/img').getall()
        # 3.数据解析
        titles = [re.findall('img.*?title="(.*?)"', each)for each in raw_url]
        images = [re.findall('img.*?src="(.*?)"', each)for each in raw_url]
        # 4.二次响应
        if not os.path.exists(destpath):
            os.mkdir(destpath)
        for image,title in zip(images,titles):
            content = r.get(image[0],headers=headers).content
            with open(os.path.join(destpath,'{}.jpg'.format(title[0])),'wb') as f:
                f.write(content)
    end = time.time()
    print('Successfully!All time:{}s'.format(int(end-start)))

baseurl = 'https://www.hexuexiao.cn/search-%E9%A3%8E%E6%99%AF?'
destpath = r'E:\学习\大3上\实验室\image/test'

if __name__ == '__main__':
    downloadImage(baseurl,destpath)