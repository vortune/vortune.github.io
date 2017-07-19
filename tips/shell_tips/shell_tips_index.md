# Shell 使用技巧

## 任务及进程管理

## 输出重定向

一般情况下，每个 Unix/Linux 命令运行时都会打开三个文件：

* 标准输入文件(stdin)：`stdin` 的文件描述符为 0，Unix 程序默认从 `stdin` 读取数据。
* 标准输出文件(stdout)：`stdout` 的文件描述符为 1，Unix 程序默认向 `stdout` 输出数据。

- 标准错误文件(stderr)：`stderr` 的文件描述符为 2，Unix 程序会向 `stderr` 流中写入错误信息。

默认情况下，command > file 将 `stdout` 重定向到 file，command < file 将 `stdin` 重定向到 file。

在终端（Terminal）中运行的程序，默认的输出对象就是但前的终端窗口。shell 可以通过输出定向符 `>` 来决定程序的输出方向。譬如，输出到一个文件：

```shell
$ echo 'hello, world!' > /tmp/output
```

也可以将输出加入到某个文件的末尾：

``` shell
$ echo 'hello, world!' >> /tmp/output
```

如果希望将 `stdout` 和 `stderr` 合并后重定向到 file，可以这样写：

``` shell
$ echo 'hello, world!' > 2>&1
```

或者

``` shell
$ echo 'hello, world!' > 1>&2
```

如果希望屏蔽 stdout 和 stderr，可以这样：

``` shell
$ echo 'hello, world!' > /dev/null 2>&1
```

### 输出重定向的应用

对于那些需要运行很长时间的程序，可以用 `&` 符号将程序设置到后台运行，并将程序的运行信息输出到一个文件当中。如：

``` shell
$ cat $LARGE_FILE > /log/log_file 2>&1 &
```

需要查看程序的运行信息时，可以使用 `tail` 命令：

``` shell
$ tail -n 10 /log/log_file
```

这个命令将不断刷新显示文件的末尾 10 行，直至按 `Ctrl + C` 终止。








