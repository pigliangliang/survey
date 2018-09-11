#author_by zhuxiaoliang

#2018-09-06 下午2:44


import  pandas as pd
from pandas.core.frame import DataFrame as df
#read category

#content = pd.read_csv('../news_type.csv',)
#print(content['数量'].head(7).values)



#read news contents
content = pd.read_csv('../marklabel0907.csv',usecols=[3,4])
content.columns = ['sentiment', 'types']
#print('contents：',content)


type_num = content.types.value_counts(dropna=False)
sentiment_num = content.sentiment.value_counts(dropna=False)


print(type_num)
#print(sentiment_num)


type1 = content.loc[content.types==1]
t1 = type1.sentiment.value_counts(dropna=False)
#print(t1.sort_index())
print(list(t1.sort_index()))

type3 = content.loc[content.types==3]
print(list(type3.sentiment.value_counts(dropna=False).sort_index()))

type4 = content.loc[content.types==4]
print(list(type4.sentiment.value_counts(dropna=False).sort_index()))

type2 = content.loc[content.types==2]
print(list(type2.sentiment.value_counts(dropna=False).sort_index()))


type5 = content.loc[content.types==5]
print(list(type5.sentiment.value_counts(dropna=False).sort_index()))


type7 = content.loc[content.types==7]
print(list(type7.sentiment.value_counts(dropna=False).sort_index()))

type6 = content.loc[content.types==6]
print(list(type6.sentiment.value_counts(dropna=False).sort_index()))

#写到文件中
import pandas as pd

write_csv = "news_statics.csv"
"""处理待存数据frame, xmin, xmax, ymin, ymax为list格式"""
frame =list(type_num.index)
xmin = list(type_num.values)
t1= list(t1.sort_index())
t3 =list(type3.sentiment.value_counts(dropna=False).sort_index())
t4 =list(type4.sentiment.value_counts(dropna=False).sort_index())
t2 =list(type2.sentiment.value_counts(dropna=False).sort_index())
t5 =list(type5.sentiment.value_counts(dropna=False).sort_index())
t7 =list(type7.sentiment.value_counts(dropna=False).sort_index())
t6 =list(type6.sentiment.value_counts(dropna=False).sort_index())
f=[t1[0],t3[0],t4[0],t2[0],t5[0],t7[0],t6[0]]
z1=[t1[1],t3[1],t4[1],t2[1],t5[1],t7[1],t6[1]]
z2=[t1[2],t3[2],t4[2],t2[2],t5[2],t7[2],t6[2]]
w= [t1[3],t3[3],t4[3],t2[3],t5[3],t7[3],t6[3]]
column_frame = pd.Series(frame, name='类别')
column_xmin = pd.Series(xmin, name='数量')
column_xmax = pd.Series(f, name='负面')
column_ymin = pd.Series(z1, name='中面')
column_ymax = pd.Series(z2, name='正面')
column_classid = pd.Series(w, name='无标签')



save = pd.DataFrame({'类别': frame, '数量': xmin, '负面': f, '中性': z1 ,'正面': z2, '无标签': w})
save.to_csv(write_csv)

