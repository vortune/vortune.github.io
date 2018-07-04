# Practical Examples of Rsync

This is an excerpt of *10 Practical Examples of Rsync command in Linux* [^1] .

## Some common options used with rsync commands

  * **-v** : verbose
  * **-r** : copies data recursively (but don’t preserve timestamps and permission while transferring data
  * **-a** : archive mode, archive mode allows copying  files recursively and it also preserves symbolic links, file  permissions, user & group ownerships and timestamps
  * **-z** : compress file data
  * **-h** : human-readable, output numbers in a human-readable format

## Copy/Sync Files and Directory Locally

### Copy/Sync a File on a Local Computer

This following command will sync a single file on a local machine  from one location to another location. Here in this example, a file name  **backup.tar** needs to be copied or synced to **/tmp/backups/** folder.

```shell
[root@tecmint]# rsync -zvh backup.tar /tmp/backups/
created directory /tmp/backups
backup.tar
sent 14.71M bytes  received 31 bytes  3.27M bytes/sec
total size is 16.18M  speedup is 1.10
```

In above example, you can see that if the destination is not already  exists rsync will create a directory automatically for destination.

### Copy/Sync a Directory on Local Computer

The following command will transfer or sync all the files of from one  directory to a different directory in the same machine. Here in this  example, **/root/rpmpkgs** contains some rpm package files and you want that directory to be copied inside **/tmp/backups/** folder.

```shell
[root@tecmint]# rsync -avzh /root/rpmpkgs /tmp/backups/
sending incremental file list
rpmpkgs/
rpmpkgs/httpd-2.2.3-82.el5.centos.i386.rpm
rpmpkgs/mod_ssl-2.2.3-82.el5.centos.i386.rpm
rpmpkgs/nagios-3.5.0.tar.gz
rpmpkgs/nagios-plugins-1.4.16.tar.gz
sent 4.99M bytes  received 92 bytes  3.33M bytes/sec
total size is 4.99M  speedup is 1.00
```

## Copy/Sync Files and Directory to or From a Server

### Copy a Directory from Local Server to a Remote Server

This command will sync a directory from a local machine to a remote machine. **For example**: There is a folder in your local computer “**rpmpkgs**” which contains some **RPM** packages and you want that local directory’s content send to a remote server, you can use following command.

```shell
[root@tecmint]$ rsync -avz rpmpkgs/ root@192.168.0.101:/home/
root@192.168.0.101's password:
sending incremental file list
./
httpd-2.2.3-82.el5.centos.i386.rpm
mod_ssl-2.2.3-82.el5.centos.i386.rpm
nagios-3.5.0.tar.gz
nagios-plugins-1.4.16.tar.gz
sent 4993369 bytes  received 91 bytes  399476.80 bytes/sec
total size is 4991313  speedup is 1.00
```

### Copy/Sync a Remote Directory to a Local Machine

This command will help you sync a remote directory to a local directory. Here in this example, a directory **/home/tarunika/rpmpkgs** which is on a remote server is being copied in your local computer in **/tmp/myrpms**.

```shell
[root@tecmint]# rsync -avzh root@192.168.0.100:/home/tarunika/rpmpkgs /tmp/myrpms
root@192.168.0.100's password:
receiving incremental file list
created directory /tmp/myrpms
rpmpkgs/
rpmpkgs/httpd-2.2.3-82.el5.centos.i386.rpm
rpmpkgs/mod_ssl-2.2.3-82.el5.centos.i386.rpm
rpmpkgs/nagios-3.5.0.tar.gz
rpmpkgs/nagios-plugins-1.4.16.tar.gz
sent 91 bytes  received 4.99M bytes  322.16K bytes/sec
total size is 4.99M  speedup is 1.00
```

## Rsync Over SSH

With rsync, we can use **SSH** (**Secure Shell**) for data transfer, using **SSH**  protocol while transferring our data you can be ensured that your data  is being transferred in a secured connection with encryption so that  nobody can read your data while it is being transferred over the wire on  the internet.

Also when we use rsync we need to provide the **user**/**root** password to accomplish that particular task, so using **SSH** option will send your logins in an encrypted manner so that your **password** will be safe.

### Copy a File from a Remote Server to a Local Server with SSH

To specify a protocol with **rsync** you need to give “**-e**” option with protocol name you want to use. Here in this example, We will be using “**ssh**” with “**-e**” option and perform data transfer.

```shell
[root@tecmint]# rsync -avzhe ssh root@192.168.0.100:/root/install.log /tmp/
root@192.168.0.100's password:
receiving incremental file list
install.log
sent 30 bytes  received 8.12K bytes  1.48K bytes/sec
total size is 30.74K  speedup is 3.77
```

### Copy a File from a Local Server to a Remote Server with SSH

```shell
[root@tecmint]# rsync -avzhe ssh backup.tar root@192.168.0.100:/backups/
root@192.168.0.100's password:
sending incremental file list
backup.tar
sent 14.71M bytes  received 31 bytes  1.28M bytes/sec
total size is 16.18M  speedup is 1.1
```

## Do a Dry Run with rsync

If you are a newbie and using rsync and don’t know what exactly your  command going do. Rsync could really mess up the things in your  destination folder and then doing an undo can be a tedious job.

**Suggested Read:** [How to Sync Two Apache Web Servers/Websites Using Rsync](https://www.tecmint.com/sync-two-apache-websites-using-rsync/)

Use of this option will not make any changes only do a dry run of the  command and shows the output of the command, if the output shows  exactly same you want to do then you can remove ‘**–dry-run**‘ option from your command and run on the terminal.

```shell
root@tecmint]# rsync --dry-run --remove-source-files -zvh backup.tar /tmp/backups/
backup.tar
sent 35 bytes  received 15 bytes  100.00 bytes/sec
total size is 16.18M  speedup is 323584.00 (DRY RUN)
```



***References :*** 

[^1]: [10 Practical Examples of Rsync Command in Linux](https://www.tecmint.com/rsync-local-remote-file-synchronization-commands/)