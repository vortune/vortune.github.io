# Vim Highlighting the Octave Code

Obtain the the vim script for octave file *.oct or *.m :

``` shell
$ git clone https://github.com/vim-scripts/octave.vim--.git
```

``` shell
$ cd octave.vim--
$ mkdir ~/.vim/syntax
$ cp octave.vim--/syntax/octave.vim ~/.vim/syntax
```

Add the following lines to your ~/.vimrc to get ViM to use the file :

```
" Octave syntax 
augroup filetypedetect 
  au! BufRead,BufNewFile *.m,*.oct set filetype=octave 
augroup END 
```

