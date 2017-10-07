# Entirely Copying a Git Folder

Some time we would like to copy a git folder between different computer, such as via USB stick. As result of Git folder copying, the information of files have changed, such as time stamp, authority and ownership...etc. In this case, Git will treats all file as new and prompts message to you adding them to the repository. Don't add the files. We just need to checkout in force.

``` shell
$ git branch
* master
$ git checkout -f master
```

 