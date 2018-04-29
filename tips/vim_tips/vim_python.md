# Turning VIM into Python Integrated Development Environment

Using VIM as python integrated development environment, have to collect various plugins of VIM, and deploy and configure them correctly.

## Auto Completion

Auto completing tool is very important for productivity of coding python.

### pydiction

[Github entry of Pydiction](https://github.com/rkulla/pydiction)

#### Installation and configuration

``` shell
$ mkdir -p ~/.vim/bundle
$ cd ~/.vim/bundle
$ git clone https://github.com/rkulla/pydiction.git
$ cp -fr pydiction/after ~/.vim
```

Confirming the structure of files likes below:

``` shell
$ tree ~/.vim
~/.vim
├── after
│   └── ftplugin
│       └── python_pydiction.vim
└── bundle
    └── pydiction/root
        └── complete-dict
```

Configuring the `~/.vimrc` . 

> **NOTICE:** Replace the location of `complete_dict` with your real home directory.

``` shell
$ touch ~/.vimrc
$ echo "filetype plugin on" >> ~/.vimrc
$ echo "let g:pydiction_location = '/home/robin/.vim/bundle/pydiction/complete-dict'" >> ~/.vimrc
$ echo "let g:pydiction_menu_height = 3" >> ~/.vimrc
```

For checking your installation and configuration, you could open python file and type `sys.` and press `tab`. pydiction will pop out the menu of possibilities:

```
sys
sys                          /home/robin/.vim/bundle/pydiction/complete-dict   
sysconf(                     /home/robin/.vim/bundle/pydiction/complete-dict   
sysconf_names                /home/robin/.vim/bundle/pydiction/complete-dict       
```

#### Updating dictionary manually

Updating `complete_dict` with new module. the dictionary that pydiction default support is very limited. So you have to update the dictionary file `complete_dict` manually. 

``` shell
$ cd ~/.vim/bundle/pydiction
$ python3 pydiction.py tensorflow
```

After that,  the original `complete_dict` has been backed up with name `complete_dict.last` . if you open `complete_dict`, you can find a lot of new dictionary item with keyword 'tensorflow' .

#### Generating dictionary item for alias

Many time, I would give a module a new name in coding practice, such as :

``` python
import tensorflow as tf
```

Ideally, a excellent auto completing tool can recognize the alias `tf` as `tensorflow`. But pydiction doesn't support this function yet. 

**IMPORTANT TIPS:** A compromise is to duplicate all of dictionary item of `tensorflow` in `complete_dict` and replace 'tensorflow' with it's alias 'tf'. We strongly recommend you backup original dictionary file:

``` shell
$ cd ~/.vim/bundle/pydiction
$ cp -f complete_dict complete_dict.orig
```

Let's try to generating the dictionary item for `np` that alias `numpy`.  **You just need one shell command:** 

```shell
grep '^numpy\.' complete-dict | sed 's/^numpy/np/g' >> complete_dict
```

## Syntax Checking

## Running Code

## Debugging

## Git Integration

