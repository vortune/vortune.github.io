# Soluations for Github Problem

## git clone or pull problem

### curl 

**curl 18**

```
Cloning into 'models'...
remote: Enumerating objects: 20, done.
remote: Counting objects: 100% (20/20), done.
remote: Compressing objects: 100% (20/20), done.
error: RPC failed; curl 18 transfer closed with outstanding read data remaining
fatal: the remote end hung up unexpectedly
fatal: early EOF
fatal: index-pack failed
```

Resolved by:

```bash
$ git config --global http.postBuffer 524288000
```

if issue don't solve, try `git clone` with `--depth=1` :

```bash
$ git clone https://github.com/vortune/vortune.github.io.git --depth=1
```

**curl 56**

```
Cloning into 'models'...
remote: Enumerating objects: 20, done.
remote: Counting objects: 100% (20/20), done.
remote: Compressing objects: 100% (20/20), done.
error: RPC failed; curl 56 GnuTLS recv error (-54): Error in the pull function.
fatal: the remote end hung up unexpectedly
fatal: early EOF
fatal: index-pack failed
```

Resolved by:

```bash
$ sudo apt install gnutls-bin
```

