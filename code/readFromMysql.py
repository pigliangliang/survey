#author_by zhuxiaoliang
#2018-08-28 下午6:59

#从mysql数据库中读取内容，写入到csv文件


def gettext(cursor,tb,field,starttime):

    sql = 'select {} from {} where notice_time > {}'.format(','.join(field),tb,starttime)
    cursor.execute(sql)
    ret = cursor.fetchall()
    for re in ret:
        print(re)


if  __name__ =="__main__":

    r_host = '172.21.0.11'
    r_user = 'root'
    r_passwd = 'renrenjinfu2018'
    r_name = 'waltz_base'
    r_port = 3306
    #conn = MySQLdb.connect(host=r_host, user=r_user, passwd=r_passwd, db=r_name, port=r_port, charset='utf8')
    #cursor = conn.cursor()
