import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
'''
def spiderHospital(url):
    response = requests.get(url)
    html=response.text
    htmlTree=bs(html,"html.parser")
    hosInfo=htmlTree.find("dl",{"class":"clearfix"})
    data={}
    for eachOne in hosInfo:
        tmpStr=str(eachOne).split("\"")
        if(tmpStr[0]!="\n"):
            urlId=tmpStr[1].split("\'")[1]
            nextUrl="https://guahao.zjol.com.cn/hospital/"+urlId
            data[tmpStr[3]]=spiderKeshi(nextUrl)
    return data
'''

def spiderKeshi(url):
    response = requests.get(url)
    html=response.text
    htmlTree=bs(html,"html.parser")
    host_info = htmlTree.find("div",{"id":"yyks-content"})
    keShi=host_info.find_all("dl")
    data={}
    for keShiInfo in keShi:
        keShiName=keShiInfo.find("dt")
        realName=re.sub('[</dt>]','',keShiName.string)
        second=keShiInfo.find_all("li")
        data[realName]={}
        for secondKs in second:
            tmpStr=str(secondKs).strip("<li><a href=\"javascript:searchByType.getScheduleByDepID")
            tmpStr=re.split('>|<',tmpStr)
            numStr=tmpStr[0].split('\'')
            secondName=tmpStr[1]
            if(len(numStr[1])>0):
                nextUrl="https://guahao.zjol.com.cn/pb/957122?deptId="+numStr[1]+"&fuzzy_deptId=0&docId=&fuzzy_docId=0"
                tmpInfo=spiderYishi(nextUrl)
                if(len(tmpInfo)>0):
                    data[realName][secondName]=tmpInfo
        
        #print(realName)
        #flobj=open("mytest.txt",'a')
        #flobj.write('%s\n'%keShiName)

    return data    

def spiderYishi(url):
    response = requests.get(url)
    html=response.text
    htmlTree=bs(html,"html.parser")
    yishi_info=htmlTree.find_all("div",{"class":"doc-txt"})
    dcArr=[]
    for yishi in yishi_info : 
        tmpStr=str(yishi).split('\'')
        if(len(tmpStr[1])>0):
            nextUrl="https://guahao.zjol.com.cn/pb/957122?deptId=&fuzzy_deptId=0&docId="+tmpStr[1]+"&fuzzy_docId=0"
            dc_info=finalPage(nextUrl)
            if(len(dc_info)>0):
                dcArr.append(dc_info)
    return dcArr



def finalPage(url):
    doctor={}
    response = requests.get(url)
    html=response.text
    htmlTree=bs(html,"html.parser")
    basic_info=htmlTree.find_all("span",{"class":"hui3-30-30 pr20"})
    if(len(basic_info)==0):
        return {}
    name_info=basic_info[0]
    gender_info=basic_info[1]
    name=re.split('>|<',str(name_info))[2]
    gender=re.split('>|<',str(gender_info))[2]

    level_info=htmlTree.find("span",{"class":"lv-14-30-2"})
    level=re.split('>|<',str(level_info))[2]

    abil_info=htmlTree.find_all("div",{"id":"more-x"})
    abil=re.split('>|<',str(abil_info))[14]
    doctor['姓名']=name
    doctor['性别']=gender
    doctor['职称']=level
    doctor['专长']=abil
    print(doctor)

    return doctor


data=spiderKeshi("https://guahao.zjol.com.cn/hospital/957122")


doctorInfo={"姓名":[],"性别":[],"职称":[],"医院":[],"一级科室":[],"二级科室":[],"专长":[]}

for fdepartment in data.items():
    for sdepartment in fdepartment[1].items():
        for doctor in sdepartment[1]:
            doctorInfo["姓名"].append(doctor["姓名"])
            doctorInfo["性别"].append(doctor["性别"])
            doctorInfo["职称"].append(doctor["职称"])
            doctorInfo["医院"].append("浙江康复医疗中心")
            doctorInfo["一级科室"].append(fdepartment[0])
            doctorInfo["二级科室"].append(sdepartment[0])
            doctorInfo["专长"].append(doctor["专长"])

finaData=pd.DataFrame(doctorInfo)
finaData.to_csv("浙江康复医疗中心.csv")

#ss="<dt>全科医疗科</dt>"
#print(re.sub('[</dt>]','',ss))
