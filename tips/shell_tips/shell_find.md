# find 命令使用技巧

有很多非 Linux 原生的文件夹，会将目录的属性设定为可写这个是有风险的，可以通过 `find` 命令将其改正过来，如果仅改变当前目录中的一层子目录的写属性：

``` shell
$ find . -maxdepth 1 -type d | xargs chmod a-x
```

也可以用 `-exec` 参数项来调用 `command` 对返回值进行处理。如果需要改变当前目录下所有深度的子目录的写属性：

```shell
$ find . -type d -exec sudo chmod a-x {} \;
```



