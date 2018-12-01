# Setting up a SSHD Server

Reference:

* [OpenSSH Server on Ubuntu](https://help.ubuntu.com/lts/serverguide/openssh-server.html)
* [SSH on Gentoo](https://wiki.gentoo.org/wiki/SSH)

## Installation and starting daemon

Installation for ssd daemon:

``` shell
$ sudo apt install openssh-server
```

After installation, we have to generate key:

``` shell
$ ssh-keygen 
...
```

Starting SSHD daemon:

``` shell
$ sudo /usr/sbin/sshd
```

> **CAUTION:** Since OpenSSH version 3.9, sshd have to run with its full path, like `/usr/sbin/sshd`, not just command name `sshd`. Otherwise the command line will be refused to run:
>
> ``` shell
> $ sudo sshd
> sshd re-exec requires execution with an absolute path
> ```

## Client login

Keygen:

``` bash
$ ssh-keygen -f "/home/robin/.ssh/known_hosts" -R "192.168.1.168"
```

logon:

``` shell
$ ssh username@192.168.1.168
```

