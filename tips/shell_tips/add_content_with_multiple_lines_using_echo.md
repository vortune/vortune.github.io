# Add Multiple Line into a File using 'echo'

Usually, we used the shell command 'echo' to send out content that has only one line to the channel I/O of unix like system. Some time we will want to 'echo' a content that has multiple line out to a file, below example is useful for you.

```shell
$ echo 'Tile of Testing
>   A content of multiple line.
> hello, world 
> 
> Bye' > a.txt
```

Don't worry about the special character, such as 'space', new line`\n`, 'tab' and etc, no necessary using escape character `\n`, `\t` and etc.