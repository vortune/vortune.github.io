# Branch management of Git

## Create a branch to track the remote branch

Inspecting all branch
```bash
$ git branch -a

* master
  remotes/origin/test
  remotes/origin/master
```
Creating a branch to track the remote branch `origin/test`
```bash
$ git checkout --track origin/test
```

or

```bash
$ git checkout -b test origin/test
```

