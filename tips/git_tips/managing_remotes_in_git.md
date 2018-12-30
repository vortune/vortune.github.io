# Managing Remote Repositories in Git

This is a documentation for quick view for managing remotes of a working Git project. See the official document [Git Basics - Working with Remotes](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) for detail.

## Inspecting Remotes 

``` shell
$ git remote
origin
$ git remote -v
origin	https://github.com/vortune/vortune.github.io.git (fetch)
origin	https://github.com/vortune/vortune.github.io.git (push)
```

## Add a Remote

We can add another remote, via `git remote add` .

``` shell
$ git remote add robin ~/tmp/vortune.github.io.git
```
And check it out :
``` shell
$ git remote -v
origin	https://github.com/vortune/vortune.github.io.git (fetch)
origin	https://github.com/vortune/vortune.github.io.git (push)
robin	/home/robin/tmp/vortune.github.io.git (fetch)
robin	/home/robin/tmp/vortune.github.io.git (push)
```

Inspect the detail of remote 'robin' :

``` shell
$ git remote show robin
* remote robin
  Fetch URL: /home/robin/tmp/vortune.github.io.git
  Push  URL: /home/robin/tmp/vortune.github.io.git
  HEAD branch: master
  Remote branches:
    local     tracked
    master    tracked
    ml_octave tracked
  Local refs configured for 'git push':
    local     pushes to local     (up to date)
    master    pushes to master    (up to date)
    ml_octave pushes to ml_octave (up to date)
```

## Pulling Something from Remote

If we want to pull something from the remote 'robin', we have to explicitly specify which branch you want to pull into your repository.

```shell
$ git branch
  local
* master
  ml_octave
```

Current branch of your working repository is 'master'. 

```shell
$ git pull robin master
From /home/robin/tmp/vortune.github.io
 * branch            master     -> FETCH_HEAD
Already up-to-date.
```

Above operation `git pull` is a shorthand for `git fetch robin master` followed by `git merge FETCH_HEAD` . See [git pull document](https://www.git-scm.com/docs/git-pull).

## Pushing some thing to Remote

Unlike only one remote, we have to explicitly specify a remote branch as pushing destination.

``` shell
$ git push robin master
Counting objects: 5, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 1.27 KiB | 0 bytes/s, done.
Total 5 (delta 1), reused 0 (delta 0)
To /home/robin/tmp/vortune.github.io.git
   e11413c..d96e6a0  master -> master
```
## Pushing a Local Branch to Remote Repository

Assume we want to push a branch named 'testbranch' to remote repo named 'origin', then :

```bash
$ git branch testbranch master
$ git push origin testbranch
```

## Deleting a Branch in Remote Repository

```bash
$ git push --delete origin testbranch
```

