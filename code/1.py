#author_by zhuxiaoliang
#2018-09-06 上午10:37


"""
import sys
while True:

    num=input()
    num_list=[]
    for i in range(int(num)):
        input_num=sys.stdin.readline()
        num_list.append(int(input_num))
        num_list=sorted(list(set(num_list)))
        for i in num_list:
            print (i)


"""

#字符串截取
"""
input1 = input()
input2 = input()

a = [input1[i:i+8] for i in range(0,len(input1),8)]
b = [input2[i:i+8] for i in range(0,len(input2),8)]
for c in a+b:
    if len(c)==8:
        print(c)
    else:
        print(c+'0'*(8-len(c)))

"""

#十六转十进制

#input = input()
#print(int(input,16))

#字符串转整数:
#10进制字符串: int('10')  ==>  10
#16进制字符串: int('10', 16)  ==>  16
#16进制字符串: int('0x10', 16)  ==>  16



#青蛙跳台阶，跳1，2，。。。。n
"""
input = int(input())

def fib(n):
    if n==0:
        return 1
    elif n==1:
        return 1
    else:
        return 2*fib(n-1)
print(fib(input))

"""
#       拼凑钱币
"""
while True:
    try:
        N=int(input())
        coins=[1,5,10,20,50,100]
        h=len(coins)
        dp=[0 for i in range(10001)]
        dp[0]=1
        for i in range(h):
            for j in range(1,N+1):
                if j>=coins[i]:
                    dp[j]+=dp[j-coins[i]]
        print (dp[N])
    except:
        break
"""


#最大矩形面积



#print(a,b,l)



"""

m = 0

temp = a
while a!=0:
    count = 0
    j = 0
    for i in l[temp-a:]:
        j+=1
        if i==0:
            count +=1
            if count == b:
                break

        for k in l[j-1:]:
           if k==1:
               j+=1
           else:
               break
    m = max(m,j)

    a -=1

print(m-1)
"""




"""
a = int(input())
d = { }
l = []
import sys
for i in range(a-1):
    k =list(map(int,input().split()))
    l.append(k)


for i in l:
    if i[0] in d.keys():
         d[i[0]].append(i[1])
    else:
         d[i[0]] =[i[1]]

count =0
for k in range(1,a+1):
    for m in d.keys():
        if m in d.values():


            count +=1
            continue


print(count)

a = eval('1'+'2')

print(type(a))
print(a)

input = list(map(str,input().split()))
temp = input
for i in input:
    temp.pop(input.index(i))
    for j in temp:
        if eval('3'+i)*2==eval(j+'2'):
            print(i,j)
    temp=input

"""
"""
input1 = list(input().split(';'))
text = input()
d ={}
for i in input1:
    d[i.split('_')[0]]=list(i.split('_')[1].split('|'))

def cut( text,d):
    dictionary = []
    maximum = 0
    for k,v in d.items():
        dictionary.extend(v)
        if len(v)>maximum:
            maximum =len(v)
    result = []
    index = len(text)
    contents = []
    #找句子中存在的实体
    while index:
        word = None
        if index - maximum <= 0:
            piece = text[:index]
            if piece in dictionary:
                word = piece
                result.append(text[:index])
                text = text[index:]
                index = len(text)
            if index == 1:
                text = text[index:]
                index = len(text)
        if word is None:
            index -= 1
    return  result

r = cut(text,d)
print(r)

-----------------------------
input = int(input())

def fib(n):
    if n<=5:
        return 1
    else:
        if n%5==0:
            return n//5
        else:
            return n//5 +1

print(fib(input))

"""


"""

import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())

    ans = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        ans.append(list(map(int, line.split())))
    for a in ans:
        if a[0]<=3:
            print(*[0,0])
        else:
            m = min(a[0]-a[1],a[1]-1)
            print(*[0,m])"""

""""
inputs = input()
cb= inputs.count('b')
cw = inputs.count('w')
if cb<=cw:
    if cb%2==0:
         print(cb+2*cb-1)
    else:
        print(2*cb+1)
else:
    if cw%2==0:
         print(cw+2*cw-1)
    else:
        print(2*cw+1)"""


'''
s = input()

max_len = 0

str_dict = {}
# 存储每次循环中最长的子串长度
one_max = 0
# 记录最近重复字符所在的位置+1
start = 0
for i in range(len(s)):
    # 判断当前字符是否在字典中和当前字符的下标是否大于等于最近重复字符的所在位置
    if s[i] in str_dict and str_dict[s[i]] >= start:
        # 记录当前字符的值+1
        start = str_dict[s[i]] + 1
    # 在此次循环中，最大的不重复子串的长度
    one_max = i - start + 1
    # 把当前位置覆盖字典中的位置
    str_dict[s[i]] = i
    # 比较此次循环的最大不重复子串长度和历史循环最大不重复子串长度
    max_len = max(max_len, one_max)
print( max_len)'''



""""
import sys
if __name__ == "__main__":
    # 读取第一行的n
    count =1
    n = int(sys.stdin.readline().strip())
    ans = []
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        ans.append(list(map(int, line.split())))

    for i in range(n-1):
        for z in  zip(ans[i],ans[i+1]):
            if z ==(1,1):
                count +=1

    print(count)
"""



"""
n =int( input())
def Fibonacci_Loop(n):
    result_list = []
    a, b = 0, 1
    while n > 0:
        result_list.append(b)
        a, b = b, a + b
        n -= 1
    return result_list
sum =0
for i in Fibonacci_Loop(n):
    if i <=n and i%2!=0:
        sum +=i
print(sum)"""


source= input()

from collections import Counter
d = dict(Counter(source))
l = []
g=[]
l=sorted(d.items(),key=lambda x:x[0])
for i in l:
    g.append(i[0])
    g.append(str(i[1]))
print(''.join(g))