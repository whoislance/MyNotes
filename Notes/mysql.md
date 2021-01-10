## mysql

- 启动

   ```bash
   sudo service mysql start
   ```

- 打开

  ```bash
  sudo mysql
  ```

- 退出

  ```sql
  quit
  ```

## database

- 创建数据库

   ```sql
   create database $database;
   ```

- 显示数据库

  ```sql
  show databses;
  ```

- 使用数据库

  ```sql
  use $database;
  ```

## table

- 创建表

   ```sql
   CREATE TABLE table_name ($column_name1 datatype, $column_name2 datatype,... $column_nameN datatype);
   ```

   ```sql
   DROP TABLE IF EXISTS `player`;
   CREATE TABLE `player`  (
     `player_id` int(11) NOT NULL AUTO_INCREMENT,
     `team_id` int(11) NOT NULL,
     `player_name` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
     `height` float(3, 2) NULL DEFAULT 0.00,
     PRIMARY KEY (`player_id`) USING BTREE,
     UNIQUE INDEX `player_name`(`player_name`) USING BTREE
   ) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;
   ```

- 表中插入数据

   ```sql
   INSERT INTO table_name ( field1, field2,...fieldN )
                          VALUES
                          ( value1, value2,...valueN );
   ```

   ```sql
   insert into player
   (player_id, team_id, player_name, height)
   values
   (04, 03, "susan", 1.67);
   ```

- 显示表

  ```sql
  show tables;
  ```

- 显示表列

  ```sql
show columns from $table;
  describe $table;
  ```

## select

- 检索不同的行

   ```sql
   select distinct $field from $table;
   ```

- 限制结果

   ```sql
   select $field from $table limit a, b;  #从第a行开始，持续b行
   ```

- 排序

   ```sql
   select $field from $table order by $field desc, $field;
   ```

## where

- 选取某个范围

  ```sql
  select $field from $table where $field between a and b;
  ```

- 空值检查

  ````sql
  select $field from $table where $field is null and $field not in (a,b,c);
  ````


## like

- 匹配任意字符

  ```sql
  select $field from $table where $field like "a%b"
  ```

- 匹配单个字符

  ```sql
  select $field from $table where $field like "a_b"
  ```

- 正则表达式

  ```sql
  select $field from $table where $field REGEXP 'abcd'
  ```

## function

- 计算

  ```
  select $field*field as $new_fiel from $table;
  ```

- 拼接字段

  ```sql
  select concat(Trim($field), 'abc', LTrim($field) as $new_field
                from $table;
  ```

- 获得当前时间

  ```
  select now();
  ```

- 字符串函数

  | function    | note              |
  | ----------- | ----------------- |
  | left()      | 返回串左边字符    |
  | right()     | 返回串右边的字符  |
  | ltrim()     | 去掉左边空格      |
  | rtrim()     | 去掉右边空格      |
  | lower()     | 小写              |
  | upper()     | 大写              |
  | locate()    | 找出串的一个字串  |
  | soundex()   | 返回串的SOUNDEX值 |
  | substring() | 返回串的子串      |
  | length()    | 返回串的长度      |

- 时间函数

  AddTime(), CurTime(), Date(), DateDiff(), Hour(), Minute(), Now(), Time()...

- 数值函数

  | function | note   |
  | -------- | ------ |
  | abs()    |        |
  | sqrt()   |        |
  | exp()    |        |
  | mod()    |        |
  | pi()     | 圆周率 |
  | rand()   | 随机数 |
  | sin()    |        |
  | cos()    |        |
  | tan()    |        |

- 汇总函数

  | function | note |
  | -------- | ---- |
  | avg()    |      |
  | count()  |      |
  | max()    |      |
  | min()    |      |
  | sum()    |      |

## group

> group by 必须出现在where子句之后，order by子句之前