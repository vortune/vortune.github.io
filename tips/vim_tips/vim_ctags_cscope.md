# Browsing C Source Code with Ctags and Cscope

Installation:

```shell
$ sudo apt install exuberant-ctags cscope vim
```

Confirm the `vim` was compiled with compatibility with *cscop*e.

```shell
$ vim --verison | grep 'cscope'
```

Set the vim search the `tags` file through upper folder until find the `tags` file.