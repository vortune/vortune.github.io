# 用 Git 进行小团队合作开发
为了表述方便，我们假设一个项目的名称，以及参与项目开发及管理的成员名称。

* 项目的代号：龙井
* 项目的成员：
  * 老张 -- 团队领导，负责项目的总体管理
  * 大李 -- 软件组组长，负责领导软件开发
  * 小明 -- 软件组工程师
  * 小芳 -- 软件组工程师

## 项目的源码链路
```mermaid
graph LR
Zhang_work(老张工作库)---Zhang_bare{老张的裸库}
Zhang_bare --- Li_work(大李的工作库)
Li_work ---Li_bare{大李的裸库}
Li_bare --- Ming_work[小明的工作库]
Li_bare --- Fang_work[小芳的工作库]
```

**原则上，每个工程师都只用自己的裸库与外界交换源代码；仅用自己该裸库对应的工作库对裸库进行操作**。



老张的工作机上，至少有两个“龙井”项目相关的 Git 库：

* 一个工作库。假设放置在他自己的家目录（$HOME）中；
* 一个裸库。假设发在目录 `/repo` 中；

老张首先创建自己的工作库：

``` shell
$ cd ~
$ mkdir longjing
$ cd longjing
$ echo '这是龙井项目的源码库' > README.md
$ git init .
$ git add .
$ git commit -m 'initial commit of project longjing'
```

