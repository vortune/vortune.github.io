# Browsing a git repo as a web service

We can use `lighttpd` as the web server.
``` bash
$ sudo apt install lighttpd
```
Then,  change directory into a git repo, and lauch the web service.
``` bash
$ git instaweb --httpd=lighttpd
```
The command will automatically lanchs the default browser and opens the links `http://127.0.0.1:1234` that is a address of the git repo.

If we want to stop the web service, just run :
``` bash
$ git instaweb --httpd=lighttpd --stop
```