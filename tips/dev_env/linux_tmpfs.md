# Linux tmpfs

Mounting a piece of memory as virtual disk:

```shell
$ sudo mount -t tmpfs -o size=2G tmpfs /mnt/tmp
```

If we always need a virtual disk, add a mounting item in the configure file `/etc/fstab`

```
tmpfs /mnt/tmp tmpfs size=2G 0 0
```

