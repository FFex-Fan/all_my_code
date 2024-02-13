import pymysql

# 连接 mysql
conn = pymysql.connect(host='localhost', user='root', passwd='20020617fan', port=3306, charset='utf8', db='unicom')
cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 基于 cursor 用于发送指令

# 发送指令(千万不要用 sql 的格式化做字符串的拼接，容易被 sql注入)
sql="update admin set password=%s where id = %s"
cursor.execute(sql, [24983248, 3, ])
conn.commit()  # 提交命令

# 关闭连接
conn.close()
cursor.close()
