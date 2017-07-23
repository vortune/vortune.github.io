# 快速编辑 Shell 命令行

《快速编辑 Shell 命令行》可能是我写的文章之中，在网络上被引用最多的了。现在离第一次发布这篇文章，不知不觉已经过去10年有多。现在连当时发布这篇文章的平台都已经不存在了，我当时是发布在 Linuxsir.org 的博客平台上的。一般的博客平台没有专门的代码渲染格式，所以那时文档的编排并不好。现在，我重新以 Markdown 格式编辑整理一下，并且调整和补充一些内容。

## 前言

想起听得最多的就是 \*nux 的初学者说最烦就是 Linux/ Unix 的命令行，所以就有了这个题目。如果你是个性急的人可以先尝试下文章结尾的综合练习体会一下 Shell 的快捷键，也许这样再看全文会更有趣。

其实，命令行适应了，可能比图形界面更有效率。至少对我来说是这样，我现在一看见那些所谓的 IDE 就有眼花缭乱感觉，真正用来写代码的面积都被挤到只有一包烟那么大了，呵呵。有时为找个选项花很长时间找对话框，也很痛苦吧。

## 为什么那么多人害怕命令行呢？

我想最大的问题就是很多人觉得命令行的输入和编辑都很“慢”，很低效。但是对于 Linux/ Unix 这类从内核得到整体架构，再到哪怕是最小的一个应用小软件都以文本来支撑的系统，没有娴熟的命令行技巧确实是玩不转的。希望本文能对你提高命令行使用效率有帮助。

但是由于 Linux/ Unix 的发行版实在是太多，Shell 的主流版本也有好几个，所以，本文所说的内容，可能和你的系统有出入，但是思想是一样的，在你自己的平台上摸索一下，你也会找到你的平台下编辑命令行的技巧和规律。

另外，Shell 的很多快捷键和 VIM, Emacs 的快捷键是相通的，所以，熟练使用 Shell 快捷键，对适应 \*nux 下的其他软件有很好的启示作用。

命令行的技巧除了本文提到的，还有很多，你可以自己慢慢积累，收集和体会。当然如果你经常需要输入很繁琐的命令，那么建议你自己写 Shell 脚本，定义 function, alias 等技巧来实现。

## 终端窗口配置的问题

当初这篇文章发布时，不同的 Linux 发行版的 Terminal 默认配置会有差别，会影响某些快捷键的效果。目前各个发行版的 Terminal 配置基本统一了。

## Shell 快捷键

编辑 Shell 命令行，就是通过在对 Shell 命令行的输入过程中，按各种快捷键来实现光标的跳转，以及进行删除和粘贴等各种编辑操作。下面就让我们逐个学习一下这些快捷键。

### 自动补齐：[Tab]

这个技巧很多人都应该会了，就是当输入命令，目录或者是文件名的时候按 [Tab] 键。系统就会帮你补齐可能要输入的东西，如果有多个选择系统会列表出来。你可以看清楚之后，再多输入一个或多个 charactor，再按 `[Tab]`。

**实验：**

``` shell
$ ec
```

按 `[Tab]`，补齐为：

``` shell
$ echo
```

### 查找和执行历史命令：[Ctrl + r], [Ctrl + p], [Ctrl + n]

在终端中按住 `[Ctrl]` 键的同时 `[r]` 键，出现提示：(reverse-i-search)，此时你尝试一下输入你以前输入过的命令，当你每输入一个字符的时候，终端都会滚动显示你的历史命令。当显示到你想找的合适的历史命令的时候，直按`[Enter]`，就执行了历史命令。如果你想清晰地确认一下这个历史命令，也可以按 `[Ctrl + e]`。这样历史命令会出以正常的 Shell 提示状态下显示，你确认了之后，再按 `[Enter]` 键执行也可以。

另外，可以按 `[Ctrl + p]` 或 `[Ctrl + n]` 快速向前或向后滚动查找一个历史命令，这对于快速提取刚刚执行过不久的命令很有用。

**实验：**

``` shell
$ echo "hello, world" [Enter]
$ hello, world
```
按 `[Ctrl + r]` 之后，接着输入 `echo` 。
```
(reverse-i-search)`ch':echo "hello,world" [Enter]
$ hello,world
```
### 取消本次命令输入：[Ctrl + c]

这个快捷键可以使你从一个可能你已经厌烦了的命令中安全地退出！！也许这是个不值一提的小技巧，但是经验告诉我它很有用。很多 Unix 初学者会习惯性地按 `[Enter]` 以摆脱困境，但是说不定就会发生灾难性的事件，譬如删除了一个重要的配置文件 :( 

### 光标跳转快捷键
为了方便大家记忆，加点英语助记语在后面 :)

* [Ctrl + a]      跳转至命令行首        Ahead of line
* [Ctrl + e]      跳转至命令行尾        End of line
* [Ctrl + f]       向前跳转一个字符    jump Forward one character
* [Ctrl + b]      向后跳转一个字符    jump Backward one character
* [Alt + f]         向前跳转到下一个字的第一个字符
* [Alt + b]        向后跳转到下一个字的第一个字符

### 编辑命令的快捷键

* [Ctrl + w]     向后删除一个字，用来对付刚刚输入的错误字很有用
* [Ctrl + u]      从光标当前位置删除所有字符至行首
* [Ctrl + k]      从光标当前位置删除所有字符至行尾
* [Ctrl + d]      删除光标当前位置的字符
* [Ctrl + y]      粘贴最后一个被删除的字
* [Alt + d]       删除从光标当前位置起，到当前字的结尾字符

## 综合练习

上面列举的快捷键，练习 2～3 天应该就能熟练，为了大家快速理解和记忆，我们来做个小小的综合练习：

**第一步：echo**
```shell
$ echo "hello, world." [Enter]
```
我们先输入 `echo"hello, world"`  这个命令，然后回车，也就看到了终端的输出：
```
$ hello, world.
```
**第二步：[Ctrl + r]**

我们试试找出历史命令 `echo"hello, world."` ，这时，我们按 `[e]`，`[c]`，`[h]` 这三个键，这个历史命令大概已经找到了。终端的显示应该是这样：
```shell
(reverse-i-search)`ech':echo "hello,world."
```
现在，如果 `[Enter]` 就会再一次执行这个命令，但我们现在来练习一下命令行的编辑。

**第三步：[Ctrl + a]**

这样，我们就取出了历史命令 `echo"hello, world."` ，并且将光标定位到行首，此时，光标应该在 `echo` 命令的 `e` 字符上高亮。终端的显示应该是这样：
```shell
$ echo "hello,world"
```

**第四步：[Alt+ d]**

删除了命令 `echo` ，并且光标仍然在行首。终端显示为：
```shell
$ "hello, world."
```

**第五步：输入命令 "printf"**

我们尝试一下用 Posix 的系统调用 printf 来替代 shell 命令 echo，输入`[p][r][i][n][t][f]` ，此时终端显示为：
```shell
$ printf "hello, world."
```
并且光标在 `f` 字符后面高亮。

**第六步：[Ctrl+ e]**

光标跳转到命令行尾部。

**第七步：[Ctrl+ b]**

光标后退一个字符，此时光标应处于后双引号 `"` 处高亮。

**第八步：输入换行转义符 "\n"**

输入 `[\][n]`，此时的终端显示应该为：
```shell
$ printf "hello, world.\n"
```
可以 `[Enter]` 执行了。

## 后记

当你熟练的时候，上面的步骤应该在 20 秒之内就完成了吧，希望大家都能成为命令行的高手！！！

如果你有如何改进的意见，请与作者联系。

​															罗峥嵘	<vortune@163.com>

​															2017.07.06，于广州。