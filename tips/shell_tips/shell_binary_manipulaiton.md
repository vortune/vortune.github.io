# Shell 二进制文件操作
Linux 上有多种二进制文件的处理工具，譬如：xxd，hexdump，dd 等等。

## xxd
xxd 命令可以方便地查看二进制文件：

``` shell
$ xxd -s 4 -l 128 $BINARY_FILE
```

其中：

* `-s` ：    是查看的起始点，如果需要从文件末尾开始查看则是 `-s -4` ；
* `-l` ：    查看内容的长度；
