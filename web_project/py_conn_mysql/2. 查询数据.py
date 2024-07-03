import pymysql

# 连接 mysql
conn = pymysql.connect(host='localhost', user='root', passwd='20020617fan', port=3306, charset='utf8', db='unicom')
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 基于 cursor 用于发送指令

# 发送指令
sql = "select * from admin where id > 2"
cursor.execute(sql)
result = cursor.fetchall() # 获取查询结果的全部数据
""" 结果均为 ====> 字典类型
    result = cursor.fetchall() # 获取查询结果的 全部数据
    result = cursor.fetchone() # 获取查询结果的 第一条数据
    result = cursor.fetchmany(几条) # 获取查询结果的 几条数据
"""
for item in result:
    print(item)

# 关闭连接
conn.close()
cursor.close()
