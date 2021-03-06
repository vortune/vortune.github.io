# Vim Tips

## vim tutorial

Vim 自带了一个教程，在 shell 提示符下运行下面的命令，即可打开。

```shell
$ vimtotor
```

该教程包含了大量的 vim 基本操作与技巧。

## vim 快捷键

* `[{`       跳转到向上一个 `{`；
* `]}`       跳转到向下一个 `}`；
* shift + zz        保存并退出；
* shift + zq       不保存并退出；
* Ctrl + g           显示当前文件路经；    

## vim 与 ctags 的互动

### 令 ctags 向上检索 tags 文件

Vim 与 ctags 结合使用进行源代码的浏览。安装好 ctags 之后，在源代码目录下运行：

``` shell
$ ctags -R .
```

这样将在源码根目录下产生 `tags` 文件，该文件记录了项目源码的所有标志。不过，当我们在下一级目录中，以 vim 编辑源码时，按 `Ctrl + ]` 键，并不能检索到对应的 `tag` 。运行如下命令：

``` shell
$ echo "set tags=tags;/" >> ~/.vimrc
```

> 注意：上面 shell 命令中的 `tags=tags;\` 不能留空格。

该命令在 vim 的资源配置文件 `~/.vimrc` 中的末尾，增加一个设置项，告诉 ctags 向上级目录中查找 `tags` 文件，直至根目录 `/`。

### Vim 中的 tags 跳转快捷键

* Ctrl + ] :    跳转至当前光标所指的单词的定义位置；
* Ctrl + o :   退回原位；
* Ctrl + t :    退回原位；
* g] :            打开当前光标所指的单词的 tag 列表。相当于 vim 的外部命令 `:tselect function_1` ；
* gd :           跳转至当前光标所指的单词的局部变量定义；
* \* :             跳转至当前光标所指的单词下一次出现的地方；
* \# :             跳转至当前光标所指的单词上一次出现的地方；

## Vim 文件浏览快捷键

* G :        跳转到文件末尾；
* gg :      跳转到文件可开头；
* H :       光标跳到页头；
* L :        光标跳到页尾；
* M :      光标跳到页中；

## 光标的跳转控制

* `'.` :  Jump to last modification line.
* ``.` : Jump to exact spot in last modification line
* 'CTRL-O' : Retrace your movements in file in backwards.
* 'CTRL-I : Retrace your movements in file in forwards.
