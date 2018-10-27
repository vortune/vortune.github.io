# Set bash prompt only displaying current directory

Almost all of Linux distribution, in the default setting of bash prompt will display the full path. It looks like below:

```bash
user@system:/folder1/folder2/folder3/folder4$
```

That maybe cause confusion in vision when we are working with a very deep directory.

The bash prompt (`stefano@linux:~$`) is only the first of a couple of prompts you might see:

- **PS1**: the default prompt you see when you open a shell

  It's value is stored in an environment variable called `PS1`. To see its value, type

  `echo $PS1`

  This will give you something like

  ```
  \[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$
  ```

  To change it, you can set a new value for the variable:

  ```
  export PS1="\u > "
  ```

  This will result in a prompt like this:

  ```
  stefano > 
  ```

- **PS2**: is your secondary prompt. This gets shown when a command is not finished. Type `echo "asd` and hit enter, the secondary prompt will let you enter more lines until you close the inverted commas.

- **PS3** is the prompt used for [`select`(2)](http://www.cl.cam.ac.uk/cgi-bin/manpage?2+select)

- **PS4** is the prompt used for `W`
- [stack traces](http://en.wikipedia.org/wiki/Stack_trace) (default: `+`)

To make the changes permanent, you append them to the end of `.bash_profile` (or `.bashrc`, see [this question](https://askubuntu.com/questions/1528/bashrc-or-bash-profile)) in your home directory.

Here's a more or less complete list of shorthand that you can use when composing these:

- **\a**     The 'bell' character
- **\A**     24h Time
- **\d**     Date (e.g. Tue Dec 21)
- **\e**     The 'escape' character
- **\h**     Hostname (up to the first ".")
- **\H**     Hostname
- **\j**     No. of jobs currently running (ps)
- **\l**     Current tty
- **\n**     Line feed
- **\t**     Time (hh:mm:ss)
- **\T**     Time (hh:mm:ss, 12h format)
- **\r**     Carriage return
- **\s**     Shell (i.e. bash, zsh, ksh..)
- **\u**     Username
- **\v**     Bash version
- **\V**     Full Bash release string
- **\w**     Current working directory
- **\W**     Last part of the current working directory
- **\!**     Current index in history
- **\#**     Command index
- **\$**     A "#" if you're root, else "$"
- **\\**     Literal Backslash
- **\@**     Time (12h format with am/pm)

You can of course insert any literal string, and any command:

```
export PS1="\u \$(pwd) > "
```

Where `$(pwd)` stands in place of "the output of" pwd.

- If the command substitution is escaped, as in `\$(pwd)`, it's evaluated every time the prompt is displayed, otherwise, as in `$(pwd)`, it's only evaluated once when bash is started.

If you want your prompt to feature colours, you can use bash's colour codes to do it. The code consists of three parts:

```
40;33;01
```

- The first part before the semicolon represents the text style.
  - 00=none
  - 01=bold
  - 04=underscore
  - 05=blink
  - 07=reverse
  - 08=concealed
- The second and third part are the colour and the background color:
  - 30=black
  - 31=red
  - 32=green
  - 33=yellow
  - 34=blue
  - 35=magenta
  - 36=cyan
  - 37=white

Each part can be omitted, assuming starting on the left. i.e. "1" means bold, "1;31" means bold and red. And you would get your terminal to print in colour by escaping the instruction with `\33[` and ending it with an `m`. 33, or 1B in hexadecimal, is the ASCII sign "ESCAPE" (a special character in the ASCII character set). Example:

```
"\33[1;31mHello World\33[m"
```

Prints "Hello World" in bright red.