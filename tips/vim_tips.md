# Vim Tips

## ctags

### 令 ctags 向上检索 tags 文件

Vim 与 ctags 结合使用进行源代码的浏览。安装好 ctags 之后，在源代码目录下运行：

``` shell
$ ctags -R .
```

这样将在源码根目录下产生 `tags` 文件，该文件记录了项目源码的所有标志。不过，当我们在下一级目录中，以 vim 编辑源码时，按 `Ctrl + ]` 键，并不能检索到对应的 `tag` 。运行如下命令：

``` shell
$ echo "set tags=tags;\" >> ~/.vimrc
```

> 注意：上面 shell 命令中的 `tags=tags;\` 不能留空格。

该命令在 vim 的资源配置文件 `~/.vimrc` 中的末尾，增加一个设置项，告诉 ctags 向上级目录中查找 `tags` 文件，直至根目录 `/`。