import pymysql

while True:
    username = input("用户名：")
    if (username.upper() == "Q"):
        break
    password = input("密码：")
    mobile = input("手机号：")
    while len(mobile) != 11:
        mobile = input("手机号不正确，请重新输入：")


    # 连接 mysql
    conn = pymysql.connect(host='localhost', user='root', passwd='20020617fan', port=3306, charset='utf8', db='unicom')
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)  # 基于 cursor 用于发送指令

    # 发送指令(千万不要用 sql 的格式化做字符串的拼接，容易被 sql注入)
    sql = "insert into admin (username,password,mobile) values(%s,%s,%s)"
    cursor.execute(sql, [username, password, mobile])  # 使用列表进行赋值
    """
        %(参数名)s  ====> 需要使用字典进行赋值
        sql = "insert into admin (username,password,mobile) values(%(n1)s,%(n2)s,%(n3)s)"
        cursor.execute(sql, {"n1":"Jack", "n2":"1221322", "n3":"15635140372"})
    """
    conn.commit()  # 提交命令，除了查询以外，其他都要执行该语句

    # 关闭连接
    conn.close()
    cursor.close()
