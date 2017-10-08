# Quickly Sharing a Git Repository

In a collaboration of small team, we hope to have a quick method to share the git repo to others. Assume you work with a laptop with wireless connectivity. The problem of sharing a git repo on wireless connectivity,  is the wireless adapters usually were assigned a dynamic IP address. 

As a result of using dynamic IP address, the Git repo may lose its host, because any machine in same local network will get a new IP address that may be different after rebooted.

## Assigning a Virtual Static IP Address

I recommend your team delivers one static IP address for every one, for example `192.168.1.28` for me. now let's to inspect the configuration of connectivity with a command line `ifconfig -a` :

``` shell
$ ifconfig -a
lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:696 errors:0 dropped:0 overruns:0 frame:0
          TX packets:696 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:78332 (78.3 KB)  TX bytes:78332 (78.3 KB)

wlp1s0    Link encap:Ethernet  HWaddr 9c:b6:d0:1d:3f:a5  
          inet addr:192.168.1.104  Bcast:192.168.1.255  Mask:255.255.255.0
          inet6 addr: fe80::2d7c:83da:9eae:83c5/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:56633 errors:0 dropped:0 overruns:0 frame:0
          TX packets:93730 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:5729296 (5.7 MB)  TX bytes:139336197 (139.3 MB)
```

We will setup a virtual device named `wlp1s0:0` that was a relevancy of `wlp1s0` , and give it a new IP address according to your teamwork's etiquette. run :

``` shell
$ sudo ifconfig wlp1s0:0 192.168.1.28
```

Checking the configuration again :

``` shell
$ ifconfig
lo        Link encap:Local Loopback  
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:714 errors:0 dropped:0 overruns:0 frame:0
          TX packets:714 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:80703 (80.7 KB)  TX bytes:80703 (80.7 KB)

wlp1s0    Link encap:Ethernet  HWaddr 9c:b6:d0:1d:3f:a5  
          inet addr:192.168.1.104  Bcast:192.168.1.255  Mask:255.255.255.0
          inet6 addr: fe80::2d7c:83da:9eae:83c5/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:56653 errors:0 dropped:0 overruns:0 frame:0
          TX packets:93762 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:5734349 (5.7 MB)  TX bytes:139339462 (139.3 MB)

wlp1s0:0  Link encap:Ethernet  HWaddr 9c:b6:d0:1d:3f:a5  
          inet addr:192.168.1.28  Bcast:192.168.1.255  Mask:255.255.255.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
```

Bingo! we have a new IP address of `192.168.1.28` .

## Exporting Base Path of Git Repositories

We assume you have a git repo named 'foo' in directory of `/home/someone/git` .

``` shell
$ ls /home/someone/git
foo bar
```

 Very easy to export the all of git repositories in the the directory :

``` shell
$ git daemon --export-all \
             --enable=receive-pack \
             --base-path=/home/someone/git /home/someone/git
```

Where:

* `--export-all` : export every git repos even it hasn't no file named 'git-export-ok' in '.git' directory.
* `--enable=receive-pack` : Allow the client to upload their work to your computer.

## Remoting Clone

Now, your partner can clone your git repo, He just runs :

``` shell
$ git clone git://192.168.1.28:/foo
```

Remember via the **virtual static IP address** !

> you can also clone the git repo via dynamic IP address of '192.168.1.104', but in that way, git repo might lost its remote host even the host rebooted.

### Inspecting Origin

Check the information of origin of the local git repo with git command `git remote` : 

``` shell
$ cd /${PATH_TO_YOUR_GIT_BASE_PATH}/foo
$ git remote show origin
* remote origin
  Fetch URL: git://192.168.1.28:/foo
  Push  URL: git://192.168.1.28:/foo
  HEAD branch: master
  Remote branch:
    master tracked
  Local branch configured for 'git pull':
    master merges with remote master
  Local ref configured for 'git push':
    master pushes to master (up to date)
```

## Collaboration

After cloning a git repo from remote host, the current git branch is a mirror of the branch master of remote repo.

``` shell
$ git branch
* master
```

### Working on Your Private Branch

Strongly recommend every don't work on branch 'master'. we should work on a new purposive branch, like :

``` shell
$ git checkout -b hotfix master
$ git branch
* hotfix
  master
```
See [Git Branching](https://git-scm.com/book/en/v1/Git-Branching) for detail.

Now, you are on branch 'hotfix'. You can do something on branch 'hotfix' and after to proof any thing clear, you are ready to merge your jobs. 

### Push Your Jobs

Before merging any thing into branch 'master', don't forget update it with remote branch.

``` shell
$ git checkout master
$ git pull
```

Above commands will merge remote branch 'master' into local branch 'master'. During the merging process, you have to resolve the conflicts between both branches if they existed. See [Basic Branching and Merging](https://git-scm.com/book/en/v1/Git-Branching-Basic-Branching-and-Merging) .

#### Merging your local jobs

Branch 'master' is alway starting point to push out anything, so you should merge any local jobs into 'master':

``` shell
$ git branch
* master
  hotfix
```

Seriously confirm 'master' is current active branch, and merge:

``` shell
$ git merge hotfix
```

#### Now ready to push

After that, you can safely push out you jobs:

``` shell
$ git push origin master
```

Enjoin Your Teamwork !




