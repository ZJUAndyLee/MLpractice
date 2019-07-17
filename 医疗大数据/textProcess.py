import pandas as pd
from textrank4zh import TextRank4Keyword, TextRank4Sentence
hospital=["浙江大学医学院附属第一医院.csv","浙江大学医学院附属第二医院.csv","浙江省肿瘤医院.csv","浙江大学医学院附属妇产科医院.csv","浙江省中医院下沙院区.csv","浙江省立同德医院.csv","浙江省新华医院(浙江中医药大学附属第二医院).csv","浙江省中山医院.csv","浙江医院.csv","浙江省人民医院.csv","浙江大学医学院附属邵逸夫医院.csv","浙江省中医院.csv","浙江省人民医院（望江院区）.csv","浙江大学儿童医院(滨江新院区).csv","浙江省口腔医院.csv","中国人民解放军第九〇三医院.csv","浙江康复医疗中心.csv"]
zhiShuData={"姓名":[],"性别":[],"职称":[],"医院":[],"一级科室":[],"二级科室":[],"简介":[]}

for hosName in hospital:
    docData=pd.read_csv(hosName)
    newData=docData.values
    for textArr in newData:
        tr4w=TextRank4Keyword()
        tr4w.analyze(text=textArr[7], lower=True, window=2)
        tmpText=[i["word"] for i in tr4w.get_keywords(8,word_min_len=1)]
        abilText=','.join(tmpText)
        zhiShuData["姓名"].append(textArr[1])
        zhiShuData["性别"].append(textArr[2])
        zhiShuData["职称"].append(textArr[3])
        zhiShuData["医院"].append(textArr[4])
        zhiShuData["一级科室"].append(textArr[5])
        zhiShuData["二级科室"].append(textArr[6])
        zhiShuData["简介"].append(abilText)

secData=pd.DataFrame(zhiShuData)
secData.to_csv("浙江省直属医院医生信息.csv")